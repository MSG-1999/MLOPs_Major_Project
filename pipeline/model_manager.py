# pipeline/model_manager.py
"""
Model Manager: Handles model loading, caching, and device management.
Supports Stable Diffusion v1.5 from HuggingFace.
"""

import os
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

import torch
from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSDEScheduler,
)
from safetensors.torch import load_file
from loguru import logger


MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"

SCHEDULER_MAP = {
    "DDIM": DDIMScheduler,
    "PNDM": PNDMScheduler,
    "LMS": LMSDiscreteScheduler,
    "Euler": EulerDiscreteScheduler,
    "Euler A": EulerAncestralDiscreteScheduler,
    "DPM++ 2M": DPMSolverMultistepScheduler,
    "DPM++ SDE": DPMSolverSDEScheduler,
}


@dataclass
class ModelInfo:
    model_id: str
    device: str
    dtype: str
    loaded_at: float = field(default_factory=time.time)
    generation_count: int = 0
    scheduler: str = "DDIM"


class ModelManager:
    """
    Singleton model manager for SD v1.5.
    Handles loading, device selection, scheduler swapping, and memory optimization.
    """

    _instance: Optional["ModelManager"] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.pipeline: Optional[StableDiffusionPipeline] = None
        self.model_info: Optional[ModelInfo] = None
        self.config: Dict[str, Any] = {}
        self.current_lora_path: Optional[str] = None
        self._load_lock = threading.Lock()

    def get_device(self, requested: str = "auto") -> str:
        """Auto-detect best available device."""
        if requested != "auto":
            return requested
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            logger.info("Apple MPS available")
        else:
            device = "cpu"
            logger.warning("No GPU detected, using CPU (will be slow!)")
        return device

    def get_dtype(self, device: str, requested: str = "float16") -> torch.dtype:
        """Determine appropriate dtype for device."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if device == "cpu":
            logger.info("CPU device: forcing float32")
            return torch.float32
        return dtype_map.get(requested, torch.float16)

    def load_model(
        self,
        model_id: str = MODEL_ID,
        device: str = "auto",
        dtype: str = "float16",
        cache_dir: str = "./model_cache",
        enable_xformers: bool = True,
        enable_attention_slicing: bool = True,
        enable_vae_slicing: bool = True,
        enable_cpu_offload: bool = False,
        safety_checker: bool = True,
        scheduler: str = "DDIM",
        **kwargs,
    ) -> "ModelManager":
        """
        Load SD v1.5 pipeline with optimizations.

        Args:
            model_id: HuggingFace model ID
            device: Target device (auto/cpu/cuda/mps)
            dtype: Model precision
            cache_dir: Local cache directory
            enable_xformers: Memory-efficient attention (requires xformers)
            enable_attention_slicing: Slice attention for lower VRAM
            enable_vae_slicing: Slice VAE for lower VRAM
            enable_cpu_offload: Offload model components to CPU
            safety_checker: Enable NSFW safety checker
            scheduler: Noise scheduler name
        """
        with self._load_lock:
            if self.pipeline is not None:
                logger.info("Model already loaded, returning cached instance")
                return self

            resolved_device = self.get_device(device)
            resolved_dtype = self.get_dtype(resolved_device, dtype)
            Path(cache_dir).mkdir(parents=True, exist_ok=True)

            logger.info(f"Loading {model_id} on {resolved_device} ({resolved_dtype})")
            start_time = time.time()

            pipeline_kwargs = {
                "pretrained_model_name_or_path": model_id,
                "torch_dtype": resolved_dtype,
                "cache_dir": cache_dir,
                # variant="fp16" tells diffusers to fetch *.fp16.safetensors files
                # instead of the default float32 .bin or .safetensors files.
                # This cuts download size from ~44 GB to ~3.4 GB by skipping:
                #   - v1-5-pruned.ckpt           (7.7 GB float32 raw checkpoint)
                #   - v1-5-pruned-emaonly.ckpt   (4.3 GB float32 EMA checkpoint)
                #   - unet/diffusion_pytorch_model.bin  (3.4 GB float32 bin)
                #   - vae, text_encoder .bin duplicates (~2 GB)
                #   - flax/tf/msgpack weights   (~22 GB)
                "variant": "fp16",
                "use_safetensors": True,
            }

            if not safety_checker:
                pipeline_kwargs["safety_checker"] = None

            try:
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    **pipeline_kwargs
                )
            except Exception as e:
                # fp16 variant not found (e.g. custom/fine-tuned model) —
                # fall back to default safetensors without variant tag
                logger.warning(
                    f"fp16 variant not available ({e}), "
                    "falling back to default safetensors"
                )
                pipeline_kwargs.pop("variant", None)
                try:
                    self.pipeline = StableDiffusionPipeline.from_pretrained(
                        **pipeline_kwargs
                    )
                except Exception as e2:
                    # Last resort: no format constraints at all
                    logger.warning(f"safetensors load failed ({e2}), trying without use_safetensors")
                    pipeline_kwargs.pop("use_safetensors", None)
                    self.pipeline = StableDiffusionPipeline.from_pretrained(
                        **pipeline_kwargs
                    )

            # Apply memory optimizations
            if enable_cpu_offload and resolved_device == "cuda":
                self.pipeline.enable_model_cpu_offload()
                logger.info("CPU offload enabled")
            else:
                self.pipeline = self.pipeline.to(resolved_device)

            if enable_attention_slicing:
                self.pipeline.enable_attention_slicing()
                logger.info("Attention slicing enabled")

            if enable_vae_slicing:
                self.pipeline.enable_vae_slicing()
                logger.info("VAE slicing enabled")

            if enable_xformers:
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("xFormers memory-efficient attention enabled")
                except Exception as e:
                    logger.warning(f"xFormers not available: {e}")

            # Set scheduler
            self.set_scheduler(scheduler)

            load_time = time.time() - start_time
            logger.success(f"Model loaded in {load_time:.1f}s")

            self.model_info = ModelInfo(
                model_id=model_id,
                device=resolved_device,
                dtype=str(resolved_dtype),
                scheduler=scheduler,
            )

            return self

    def set_scheduler(self, scheduler_name: str):
        """Hot-swap the noise scheduler."""
        if self.pipeline is None:
            raise RuntimeError("Model not loaded")

        scheduler_cls = SCHEDULER_MAP.get(scheduler_name)
        if scheduler_cls is None:
            raise ValueError(f"Unknown scheduler: {scheduler_name}. Choose from {list(SCHEDULER_MAP.keys())}")

        self.pipeline.scheduler = scheduler_cls.from_config(
            self.pipeline.scheduler.config
        )

        if self.model_info:
            self.model_info.scheduler = scheduler_name

        logger.info(f"Scheduler set to: {scheduler_name}")

    def is_loaded(self) -> bool:
        return self.pipeline is not None

    def get_model_info(self) -> Optional[ModelInfo]:
        return self.model_info

    def get_vram_usage(self) -> Dict[str, float]:
        """Get current VRAM usage in MB."""
        if not torch.cuda.is_available():
            return {"allocated_mb": 0, "reserved_mb": 0, "total_mb": 0}
        return {
            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
            "total_mb": torch.cuda.get_device_properties(0).total_memory / 1024**2,
        }

    def load_lora(self, lora_path: str):
        """Apply LoRA weights to the current pipeline."""
        if self.pipeline is None:
            raise RuntimeError("Model not loaded")

        logger.info(f"Loading LoRA from {lora_path}")
        try:
            # Check if it's a directory (PEFT-style) or a single file
            weights_path = lora_path
            if os.path.isdir(lora_path):
                # Standard PEFT names
                for filename in ["adapter_model.safetensors", "pytorch_lora_weights.safetensors"]:
                    candidate = os.path.join(lora_path, filename)
                    if os.path.exists(candidate):
                        weights_path = candidate
                        break
            
            if weights_path.endswith(".safetensors"):
                state_dict = load_file(weights_path)
                
                # Strip PEFT prefix if present
                # PEFT wraps the model in 'base_model.model', which diffusers doesn't expect
                new_state_dict = {}
                prefix = "base_model.model."
                prefix_found = False
                
                for k, v in state_dict.items():
                    if k.startswith(prefix):
                        new_state_dict[k[len(prefix):]] = v
                        prefix_found = True
                    else:
                        new_state_dict[k] = v
                
                if prefix_found:
                    logger.info("Detected and stripped 'base_model.model.' prefix from LoRA state dict")
                    self.pipeline.load_lora_weights(new_state_dict)
                else:
                    # No prefix, load normally
                    self.pipeline.load_lora_weights(lora_path)
            else:
                # Fallback for .bin or other formats
                self.pipeline.load_lora_weights(lora_path)

            self.current_lora_path = lora_path
            logger.success("LoRA weights loaded")
        except Exception as e:
            logger.error(f"Failed to load LoRA: {e}")
            raise

    def unload_lora(self):
        """Revert the pipeline to base model."""
        if self.pipeline is None:
            return

        if self.current_lora_path:
            logger.info("Unloading LoRA weights")
            self.pipeline.unload_lora_weights()
            self.current_lora_path = None
            logger.success("LoRA weights unloaded")

    def unload_model(self):
        """Free GPU memory."""
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
            self.model_info = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model unloaded and memory freed")
