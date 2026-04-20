# pipeline/inference_engine.py
"""
Inference Engine: Core generation logic with full parameter control,
seed management, and result packaging.
"""

import time
import uuid
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Callable, Any

import torch
import numpy as np
from PIL import Image
from loguru import logger

from pipeline.model_manager import ModelManager
from pipeline.prompt_processor import PromptProcessor, ProcessedPrompt


@dataclass
class GenerationConfig:
    """All parameters for a single generation request."""
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    num_images: int = 1
    scheduler: str = "DDIM"
    style_preset: str = "None"
    negative_preset: str = "Default"
    quality_boost: bool = False
    # Metadata
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class GenerationResult:
    """Result from a generation request."""
    request_id: str
    images: List[Image.Image]
    prompt_used: str
    negative_prompt_used: str
    config: GenerationConfig
    generation_time_s: float
    seed_used: int
    nsfw_detected: bool = False
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    @property
    def success(self) -> bool:
        return self.error is None and len(self.images) > 0


class InferenceEngine:
    """
    Core inference engine for Stable Diffusion v1.5.
    Wraps the pipeline with error handling, seed management, and callbacks.
    """

    def __init__(self):
        self.model_manager = ModelManager()
        self.prompt_processor = PromptProcessor()
        self._generation_count = 0

    def _resolve_seed(self, seed: Optional[int]) -> int:
        """Get deterministic or random seed."""
        if seed is None or seed == -1:
            return random.randint(0, 2**32 - 1)
        return seed

    def _get_generator(self, seed: int, device: str) -> torch.Generator:
        """Create seeded torch generator."""
        if device == "cpu" or not torch.cuda.is_available():
            return torch.Generator(device="cpu").manual_seed(seed)
        return torch.Generator(device=device).manual_seed(seed)

    def generate(
        self,
        config: GenerationConfig,
        progress_callback: Optional[Callable[[int, int, Any], None]] = None,
    ) -> GenerationResult:
        """
        Run the full generation pipeline.

        Args:
            config: GenerationConfig with all parameters
            progress_callback: Optional fn(step, total_steps, latent) called each step

        Returns:
            GenerationResult with images and metadata
        """
        if not self.model_manager.is_loaded():
            raise RuntimeError("Model not loaded. Call model_manager.load_model() first.")

        # Apply scheduler if different
        if (self.model_manager.model_info and
                self.model_manager.model_info.scheduler != config.scheduler):
            self.model_manager.set_scheduler(config.scheduler)

        # Process prompts
        processed: ProcessedPrompt = self.prompt_processor.process(
            prompt=config.prompt,
            negative_prompt=config.negative_prompt,
            style_preset=config.style_preset,
            negative_preset=config.negative_preset,
            quality_boost=config.quality_boost,
        )

        seed = self._resolve_seed(config.seed)
        device = self.model_manager.model_info.device
        generator = self._get_generator(seed, device)

        logger.info(
            f"[{config.request_id}] Generating {config.num_images}x image(s) "
            f"| {config.width}x{config.height} | {config.num_inference_steps} steps "
            f"| CFG {config.guidance_scale} | seed {seed}"
        )

        start_time = time.time()
        nsfw_detected = False
        images = []

        try:
            pipeline = self.model_manager.pipeline

            # Build callback for progress tracking
            def step_callback(step: int, timestep: int, latents: Any):
                if progress_callback:
                    progress_callback(step, config.num_inference_steps, latents)

            pipeline_output = pipeline(
                prompt=processed.enhanced,
                negative_prompt=processed.negative if processed.negative else None,
                width=config.width,
                height=config.height,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                num_images_per_prompt=config.num_images,
                generator=generator,
                callback=step_callback if progress_callback else None,
                callback_steps=1 if progress_callback else 1,
            )

            images = pipeline_output.images

            # Check for NSFW flags
            if hasattr(pipeline_output, "nsfw_content_detected"):
                nsfw_flags = pipeline_output.nsfw_content_detected
                if nsfw_flags and any(nsfw_flags):
                    nsfw_detected = True
                    logger.warning(f"[{config.request_id}] NSFW content detected")
                    # Replace flagged images with black
                    images = [
                        Image.new("RGB", img.size, (0, 0, 0)) if flagged else img
                        for img, flagged in zip(images, nsfw_flags)
                    ]

            generation_time = time.time() - start_time
            self._generation_count += 1
            if self.model_manager.model_info:
                self.model_manager.model_info.generation_count += 1

            logger.success(
                f"[{config.request_id}] Generated in {generation_time:.2f}s "
                f"({config.num_inference_steps / generation_time:.1f} steps/s)"
            )

            return GenerationResult(
                request_id=config.request_id,
                images=images,
                prompt_used=processed.enhanced,
                negative_prompt_used=processed.negative,
                config=config,
                generation_time_s=generation_time,
                seed_used=seed,
                nsfw_detected=nsfw_detected,
            )

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"[{config.request_id}] CUDA OOM: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return GenerationResult(
                request_id=config.request_id,
                images=[],
                prompt_used=processed.enhanced,
                negative_prompt_used=processed.negative,
                config=config,
                generation_time_s=time.time() - start_time,
                seed_used=seed,
                error=f"GPU out of memory. Try reducing image size or steps. ({e})",
            )

        except Exception as e:
            logger.error(f"[{config.request_id}] Generation failed: {e}")
            return GenerationResult(
                request_id=config.request_id,
                images=[],
                prompt_used=processed.enhanced,
                negative_prompt_used=processed.negative,
                config=config,
                generation_time_s=time.time() - start_time,
                seed_used=seed,
                error=str(e),
            )

    def get_total_generations(self) -> int:
        return self._generation_count
