#!/usr/bin/env python3
# scripts/download_model.py
"""
Pre-download SD v1.5 fp16 safetensors weights to local cache (~3.4 GB).

WHY THIS IS SMALL:
  The SD v1.5 repo contains many redundant formats:
    - v1-5-pruned.ckpt          7.7 GB  (raw float32 — NOT needed by diffusers)
    - v1-5-pruned-emaonly.ckpt  4.3 GB  (another raw checkpoint — NOT needed)
    - unet/diffusion_pytorch_model.bin  3.4 GB  (float32 bin — we use fp16 safetensors)
    - vae/, text_encoder/ .bin files    ~2 GB   (more float32 duplicates)
    - flax_model*, tf_model*, msgpack   ~10 GB  (other framework weights)

  We use allow_patterns (whitelist) to grab ONLY the fp16 safetensors
  variant files that diffusers actually needs, totalling ~3.4 GB:
    - unet/diffusion_pytorch_model.fp16.safetensors       ~1.7 GB
    - vae/diffusion_pytorch_model.fp16.safetensors        ~320 MB
    - text_encoder/model.fp16.safetensors                 ~235 MB
    - safety_checker/model.fp16.safetensors               ~580 MB
    - tokenizer/, scheduler/, feature_extractor/ configs  ~1 MB
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"

# Whitelist: only fetch fp16 safetensors + tiny config/tokenizer files.
# Everything else (float32 .bin, raw .ckpt, flax, tf) is excluded by default.
ALLOW_PATTERNS = [
    # fp16 safetensors weights (the only files we load at runtime)
    "*.fp16.safetensors",
    # JSON configs for every sub-component
    "**/*.json",
    # Tokenizer vocabulary files
    "tokenizer/vocab.json",
    "tokenizer/merges.txt",
    "tokenizer_config.json",
    # Feature extractor preprocessor
    "feature_extractor/preprocessor_config.json",
]

# Belt-and-suspenders blocklist — catches anything the whitelist misses
IGNORE_PATTERNS = [
    # Raw monolithic checkpoints (7.7 GB + 4.3 GB)
    "*.ckpt",
    # Float32 PyTorch bin weights (we use fp16 safetensors instead)
    "diffusion_pytorch_model.bin",
    "pytorch_model.bin",
    "model.bin",
    # Other framework weights
    "*.msgpack",
    "*.h5",
    "flax_model*",
    "tf_model*",
    "rust_model.ot",
    "tf_weights*",
]


def show_size_breakdown():
    print("\n  What you're downloading (~3.4 GB total):")
    print("  ├─ unet/diffusion_pytorch_model.fp16.safetensors  ~1.7 GB")
    print("  ├─ safety_checker/model.fp16.safetensors          ~580 MB")
    print("  ├─ vae/diffusion_pytorch_model.fp16.safetensors   ~320 MB")
    print("  ├─ text_encoder/model.fp16.safetensors            ~235 MB")
    print("  └─ configs, tokenizer, scheduler files            ~1 MB")
    print("\n  What's being skipped (saves ~40 GB):")
    print("  ✗  v1-5-pruned.ckpt               7.7 GB  (float32 raw checkpoint)")
    print("  ✗  v1-5-pruned-emaonly.ckpt        4.3 GB  (float32 EMA checkpoint)")
    print("  ✗  unet/diffusion_pytorch_model.bin 3.4 GB (float32 bin duplicate)")
    print("  ✗  vae, text_encoder .bin files    ~2 GB   (float32 bin duplicates)")
    print("  ✗  flax_model*, tf_model* files    ~22 GB  (other framework weights)\n")


def download():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface-hub")
        sys.exit(1)

    try:
        from loguru import logger
    except ImportError:
        import logging
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    CACHE_DIR = Path("./model_cache")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Downloading: {MODEL_ID}")
    print(f"  Cache dir:   {CACHE_DIR.resolve()}")
    print(f"  Strategy:    fp16 safetensors only (whitelist)")
    print(f"{'='*60}")
    show_size_breakdown()

    start = time.time()

    try:
        local_dir = snapshot_download(
            repo_id=MODEL_ID,
            cache_dir=str(CACHE_DIR),
            allow_patterns=ALLOW_PATTERNS,
            ignore_patterns=IGNORE_PATTERNS,
        )
        elapsed = time.time() - start
        logger.success(
            f"Download complete in {elapsed:.0f}s  |  "
            f"Cached at: {local_dir}"
        )
        print(f"\n✅ Done in {elapsed/60:.1f} min — ready to generate images!")
        print(f"   Run: streamlit run app/main.py\n")

    except Exception as e:
        logger.error(f"Download failed: {e}")
        print(f"\n❌ Download failed: {e}")
        print("   Try: huggingface-cli login  (if model requires authentication)")
        sys.exit(1)


if __name__ == "__main__":
    download()
