"""
pipeline/lora_trainer.py
========================
LoRA Fine-tuning Engine for Stable Diffusion v1.5
Dataset: Local data prepared by data.py (images + metadata.jsonl)

Pipeline
────────
1. Load train / val / test splits from local 'data/' folder
2. Build LoRA adapters on UNet (PEFT)
3. Optuna HPO search for best hyper-parameters
4. Retrain best config and save weights
"""

import os
import time
import argparse
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import yaml
import optuna
from optuna.pruners import SuccessiveHalvingPruner
from loguru import logger

from datasets import load_dataset

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_cosine_schedule_with_warmup

from transformers import CLIPTextModel, CLIPTokenizer

from peft import LoraConfig, get_peft_model


# ======================================================================
# CONSTANTS
# ======================================================================

BASE_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"

DEFAULT_TARGET_MODULES = [
    "to_q", "to_k", "to_v", "to_out.0",
    "ff.net.0.proj", "ff.net.2",
]


# ======================================================================
# CONFIG
# ======================================================================

@dataclass
class LoRATrainingConfig:
    """All hyper-parameters for one training run."""

    # Data
    data_dir:          str           = "data"
    image_size:        int           = 512
    center_crop:       bool          = True
    random_flip:       bool          = True
    max_train_samples: Optional[int] = None

    # LoRA
    lora_rank:      int       = 4
    lora_alpha:     int       = 32
    lora_dropout:   float     = 0.1
    target_modules: List[str] = field(default_factory=lambda: list(DEFAULT_TARGET_MODULES))

    # Optimiser
    learning_rate:   float = 1e-4
    warmup_ratio:    float = 0.05
    mixed_precision: str   = "fp16"   # "fp16" | "bf16" | "no"

    # Training
    num_epochs:                  int   = 5
    train_batch_size:            int   = 4
    gradient_accumulation_steps: int   = 2
    noise_offset:                float = 0.05

    # I/O
    output_dir:        str = "./lora_weights"
    cache_dir:         str = "./model_cache"
    log_every_n_steps: int = 20

    # MLflow (optional)
    mlflow_tracking_uri: str = "http://localhost:5012"
    mlflow_experiment:   str = "sd-lora-indian-festivals"

    @property
    def amp_dtype(self) -> torch.dtype:
        if self.mixed_precision == "fp16":
            return torch.float16
        if self.mixed_precision == "bf16":
            return torch.bfloat16
        return torch.float32

    @property
    def use_amp(self) -> bool:
        return self.mixed_precision in ("fp16", "bf16")


def load_config(config_path: str = "configs/config.yaml") -> LoRATrainingConfig:
    with open(config_path) as f:
        full = yaml.safe_load(f)

    train  = full.get("training", {})
    params = train.get("params", {})
    lora   = train.get("lora", {})
    mlops  = full.get("mlops", {}).get("mlflow", {})

    return LoRATrainingConfig(
        data_dir                    = train.get("data_dir", "data"),
        output_dir                  = train.get("output_dir", "./lora_weights"),
        lora_rank                   = int(lora.get("rank", 4)),
        lora_alpha                  = int(lora.get("alpha", 32)),
        lora_dropout                = float(lora.get("dropout", 0.1)),
        target_modules              = lora.get("target_modules", DEFAULT_TARGET_MODULES),
        learning_rate               = float(params.get("learning_rate", 1e-4)),
        warmup_ratio                = float(params.get("warmup_ratio", 0.05)),
        num_epochs                  = int(params.get("epochs", 5)),
        train_batch_size            = int(params.get("batch_size", 4)),
        mixed_precision             = params.get("mixed_precision", "fp16"),
        gradient_accumulation_steps = int(params.get("gradient_accumulation_steps", 2)),
        log_every_n_steps           = int(params.get("log_every_n_steps", 20)),
        mlflow_tracking_uri         = mlops.get("tracking_uri", "http://localhost:5012"),
    )


# ======================================================================
# DATASET
# ======================================================================

class LoRADataset(Dataset):
    """Wraps an HF ImageFolder dataset with augmentation + tokenisation."""

    def __init__(
        self,
        hf_dataset,
        tokenizer: CLIPTokenizer,
        image_size: int  = 512,
        center_crop: bool = True,
        random_flip: bool = True,
    ):
        self.data      = hf_dataset
        self.tokenizer = tokenizer

        aug = [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size) if center_crop else transforms.RandomCrop(image_size),
        ]
        if random_flip:
            aug.append(transforms.RandomHorizontalFlip())
        aug += [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
        self.transform = transforms.Compose(aug)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        sample = self.data[idx]
        img    = sample["image"]
        prompt = sample["text"]          # from metadata.jsonl

        if img.mode != "RGB":
            img = img.convert("RGB")

        pixel_values = self.transform(img)
        token_out = self.tokenizer(
            prompt,
            padding    = "max_length",
            max_length = self.tokenizer.model_max_length,
            truncation = True,
            return_tensors = "pt",
        )

        return {
            "pixel_values": pixel_values,
            "input_ids":    token_out.input_ids.squeeze(0),
            "prompt":       prompt,
        }


def build_dataloader(
    cfg: LoRATrainingConfig,
    tokenizer: CLIPTokenizer,
    split: str = "train",
) -> DataLoader:
    """
    Loads one split (train / val / test) from data/<split>/ and returns
    a DataLoader. Augmentation is enabled only for the training split.
    """
    data_path = Path(cfg.data_dir) / split
    if not data_path.exists():
        raise FileNotFoundError(f"Data split not found: {data_path}")

    logger.info(f"Loading '{split}' split from: {data_path}")
    hf_ds = load_dataset(
        "imagefolder",
        data_dir  = str(data_path),
        split     = "train",        # imagefolder always labels its split "train"
        cache_dir = cfg.cache_dir,
    )

    if split == "train" and cfg.max_train_samples and cfg.max_train_samples < len(hf_ds):
        hf_ds = hf_ds.select(range(cfg.max_train_samples))
        logger.info(f"Truncated train split to {cfg.max_train_samples} samples.")

    is_train = (split == "train")
    dataset  = LoRADataset(
        hf_ds,
        tokenizer,
        image_size  = cfg.image_size,
        center_crop = cfg.center_crop,
        random_flip = cfg.random_flip and is_train,   # no augmentation on val/test
    )

    return DataLoader(
        dataset,
        batch_size  = cfg.train_batch_size,
        shuffle     = is_train,
        num_workers = 2,
        pin_memory  = torch.cuda.is_available(),
        drop_last   = is_train,
    )


# ======================================================================
# MODEL UTILS
# ======================================================================

def _resolve_model_root(cache_dir: str) -> str:
    """Returns the local snapshot path if cached, otherwise the Hub ID."""
    slug      = "models--stable-diffusion-v1-5--stable-diffusion-v1-5"
    snaps_dir = Path(cache_dir) / slug / "snapshots"
    if snaps_dir.exists():
        snaps = sorted(snaps_dir.iterdir(), key=lambda p: p.stat().st_mtime)
        if snaps:
            return str(snaps[-1])
    return BASE_MODEL_ID


def load_frozen_components(cfg: LoRATrainingConfig, device: str):
    """Loads VAE, text encoder, tokeniser, and noise scheduler — all frozen."""
    root = _resolve_model_root(cfg.cache_dir)

    vae = AutoencoderKL.from_pretrained(
        root, subfolder="vae", cache_dir=cfg.cache_dir
    ).to(device, dtype=cfg.amp_dtype)
    vae.requires_grad_(False)

    text_encoder = CLIPTextModel.from_pretrained(
        root, subfolder="text_encoder", cache_dir=cfg.cache_dir
    ).to(device, dtype=cfg.amp_dtype)
    text_encoder.requires_grad_(False)

    tokenizer       = CLIPTokenizer.from_pretrained(root, subfolder="tokenizer", cache_dir=cfg.cache_dir)
    noise_scheduler = DDPMScheduler.from_pretrained(root, subfolder="scheduler",  cache_dir=cfg.cache_dir)

    return vae, text_encoder, tokenizer, noise_scheduler


def build_lora_unet(cfg: LoRATrainingConfig, device: str) -> UNet2DConditionModel:
    root = _resolve_model_root(cfg.cache_dir)
    unet = UNet2DConditionModel.from_pretrained(
        root, subfolder="unet", cache_dir=cfg.cache_dir
    ).to(device)
    unet.requires_grad_(False)

    lora_cfg = LoraConfig(
        r              = cfg.lora_rank,
        lora_alpha     = cfg.lora_alpha,
        target_modules = cfg.target_modules,
        lora_dropout   = cfg.lora_dropout,
        bias           = "none",
    )
    unet = get_peft_model(unet, lora_cfg)
    unet.enable_gradient_checkpointing()
    unet.train()
    return unet


# ======================================================================
# LOSS
# ======================================================================

def compute_loss(
    batch,
    unet,
    vae,
    text_encoder,
    noise_scheduler,
    device: str,
    noise_offset: float = 0.05,
) -> torch.Tensor:
    weight_dtype = next(vae.parameters()).dtype
    pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
    input_ids    = batch["input_ids"].to(device)

    with torch.no_grad():
        latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor

    noise = torch.randn_like(latents)
    if noise_offset > 0:
        noise = noise + noise_offset * torch.randn(
            latents.shape[0], latents.shape[1], 1, 1,
            device=device, dtype=weight_dtype,
        )

    bsz       = latents.shape[0]
    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device
    ).long()
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    with torch.no_grad():
        encoder_hidden_states = text_encoder(input_ids)[0]

    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
    return F.mse_loss(model_pred.float(), noise.float(), reduction="mean")


# ======================================================================
# TRAINER
# ======================================================================

class LoRATrainer:
    def __init__(self, cfg: LoRATrainingConfig):
        self.cfg    = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Device: {self.device} | Mixed precision: {cfg.mixed_precision}")

    # ------------------------------------------------------------------
    def _run_eval(self, unet, vae, text_encoder, noise_scheduler, loader) -> float:
        """Single pass over a dataloader — returns mean loss."""
        unet.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                loss = compute_loss(
                    batch, unet, vae, text_encoder,
                    noise_scheduler, self.device, self.cfg.noise_offset,
                )
                total += loss.item()
                n     += 1
        unet.train()
        return total / max(n, 1)

    # ------------------------------------------------------------------
    def train(self, trial: optuna.Trial = None) -> dict:
        cfg    = self.cfg
        device = self.device

        vae, text_encoder, tokenizer, noise_scheduler = load_frozen_components(cfg, device)
        unet = build_lora_unet(cfg, device)

        train_loader = build_dataloader(cfg, tokenizer, split="train")
        val_loader   = build_dataloader(cfg, tokenizer, split="val")

        steps_per_epoch = len(train_loader) // cfg.gradient_accumulation_steps
        total_steps     = steps_per_epoch * cfg.num_epochs
        warmup_steps    = max(1, int(total_steps * cfg.warmup_ratio))

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, unet.parameters()),
            lr=cfg.learning_rate,
        )
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps   = warmup_steps,
            num_training_steps = total_steps,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

        global_step   = 0
        best_val_loss = float("inf")
        train_loss    = 0.0
        start_time    = time.time()

        for epoch in range(cfg.num_epochs):
            unet.train()
            for step, batch in enumerate(train_loader):
                with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                    loss = compute_loss(
                        batch, unet, vae, text_encoder,
                        noise_scheduler, device, cfg.noise_offset,
                    )

                scaler.scale(loss / cfg.gradient_accumulation_steps).backward()

                if (step + 1) % cfg.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    train_loss   = loss.item()

                    if global_step % cfg.log_every_n_steps == 0:
                        logger.info(
                            f"Epoch {epoch+1}/{cfg.num_epochs} | "
                            f"Step {global_step}/{total_steps} | "
                            f"Loss {train_loss:.4f} | "
                            f"LR {lr_scheduler.get_last_lr()[0]:.2e}"
                        )
                        if trial is not None:
                            trial.report(train_loss, global_step)
                            if trial.should_prune():
                                raise optuna.TrialPruned()

            # ── Validation ────────────────────────────────────────────
            val_loss = self._run_eval(unet, vae, text_encoder, noise_scheduler, val_loader)
            logger.info(f"Epoch {epoch+1} | Val loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save(unet, tag="best")
                logger.info("  ↳ New best — checkpoint saved.")

        self._save(unet, tag=f"final_{int(time.time())}")

        return {
            "train_loss":    train_loss,
            "best_val_loss": best_val_loss,
            "total_time_s":  round(time.time() - start_time, 1),
        }

    # ------------------------------------------------------------------
    def evaluate_test(self) -> None:
        """Load best checkpoint and report test-set loss."""
        vae, text_encoder, tokenizer, noise_scheduler = load_frozen_components(self.cfg, self.device)
        unet = build_lora_unet(self.cfg, self.device)

        best_path = Path(self.cfg.output_dir) / "best"
        if best_path.exists():
            from peft import PeftModel
            unet = PeftModel.from_pretrained(unet, str(best_path))
            logger.info(f"Loaded best checkpoint from {best_path}")
        else:
            logger.warning("No 'best' checkpoint found — using freshly initialised LoRA weights.")

        test_loader = build_dataloader(self.cfg, tokenizer, split="test")
        test_loss   = self._run_eval(unet, vae, text_encoder, noise_scheduler, test_loader)
        logger.info(f"Test loss: {test_loss:.4f}")

    # ------------------------------------------------------------------
    def _save(self, unet, tag: str) -> str:
        save_path = Path(self.cfg.output_dir) / tag
        save_path.mkdir(parents=True, exist_ok=True)
        unet.save_pretrained(str(save_path))
        logger.info(f"Checkpoint saved → {save_path}")
        return str(save_path)


# ======================================================================
# HPO
# ======================================================================

def hpo_objective(trial: optuna.Trial, base_cfg: LoRATrainingConfig) -> float:
    cfg = replace(
        base_cfg,
        lora_rank        = trial.suggest_categorical("lora_rank",    [4, 8, 16]),
        lora_alpha       = trial.suggest_categorical("lora_alpha",   [16, 32, 64]),
        lora_dropout     = trial.suggest_float("lora_dropout",       0.0, 0.2),
        learning_rate    = trial.suggest_float("learning_rate",      1e-5, 2e-4, log=True),
        train_batch_size = trial.suggest_categorical("train_batch_size", [1, 2, 4]),
        num_epochs       = 1,   # short runs for HPO
    )
    trainer = LoRATrainer(cfg)
    try:
        result = trainer.train(trial)
        return result["best_val_loss"]
    except optuna.TrialPruned:
        raise
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        return float("inf")


# ======================================================================
# ENTRY POINT
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tuner for Stable Diffusion v1.5")
    parser.add_argument(
        "command", choices=["train", "hpo", "test"],
        help="train: full fine-tune | hpo: hyper-param search | test: eval on test split",
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--trials", type=int, default=5, help="Optuna trial count (hpo only)")
    args = parser.parse_args()

    # ── Logging ───────────────────────────────────────────────────────
    os.makedirs("logs", exist_ok=True)
    logger.add("logs/training.log", rotation="10 MB", level="INFO", enqueue=True)
    logger.info(f"PID {os.getpid()} | command={args.command}")

    # ── Config ────────────────────────────────────────────────────────
    base_cfg = load_config(args.config)

    # ── Dispatch ──────────────────────────────────────────────────────
    if args.command == "train":
        trainer = LoRATrainer(base_cfg)
        result  = trainer.train()
        logger.info(f"Done: {result}")

    elif args.command == "hpo":
        study = optuna.create_study(
            direction = "minimize",
            sampler   = optuna.samplers.TPESampler(seed=42),
            pruner    = SuccessiveHalvingPruner(),
        )
        study.optimize(lambda t: hpo_objective(t, base_cfg), n_trials=args.trials)
        logger.info(f"Best trial #{study.best_trial.number}: {study.best_trial.params}")
        logger.info(f"Best val loss: {study.best_value:.4f}")

    elif args.command == "test":
        trainer = LoRATrainer(base_cfg)
        trainer.evaluate_test()


if __name__ == "__main__":
    main()