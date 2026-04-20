# pipeline/prompt_processor.py
"""
Prompt Processor: Validates, enhances, and preprocesses prompts
for better Stable Diffusion outputs.
"""

import re
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from loguru import logger


# Quality boosting tokens that generally improve SD outputs
QUALITY_BOOSTERS = [
    "masterpiece", "best quality", "ultra detailed", "sharp focus",
    "high resolution", "8k uhd", "photorealistic"
]

# Style modifiers grouped by category
STYLE_PRESETS = {
    "Photorealistic": "photorealistic, ultra-detailed, sharp focus, DSLR photography, 8k uhd",
    "Oil Painting": "oil painting, impasto technique, textured brushstrokes, fine art",
    "Watercolor": "watercolor painting, soft edges, transparent washes, artistic",
    "Digital Art": "digital art, vibrant colors, concept art, trending on artstation",
    "Anime": "anime style, cel shading, vibrant, studio ghibli inspired",
    "Sketch": "pencil sketch, detailed linework, crosshatching, graphite",
    "Cinematic": "cinematic lighting, film still, anamorphic lens, bokeh",
    "Fantasy": "fantasy art, magical, ethereal, epic, dramatic lighting",
    "None": "",
}

# Common negative prompt templates
NEGATIVE_PRESETS = {
    "Default": "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face, blurry, draft",
    "Photorealistic": "cartoon, illustration, painting, drawing, 3d render, anime, manga, cgi, artificial, fake, low quality, blurry, noise",
    "Artistic": "photograph, photo, realistic, ugly, deformed, noisy, low quality, blurry",
    "Minimal": "low quality, blurry, watermark",
    "None": "",
}


@dataclass
class ProcessedPrompt:
    original: str
    enhanced: str
    negative: str
    style_applied: str
    quality_boost: bool
    token_count: int
    warnings: List[str] = field(default_factory=list)


class PromptProcessor:
    """
    Processes and enhances text prompts for Stable Diffusion generation.
    """

    MAX_TOKENS = 77  # CLIP token limit
    MAX_CHARS = 400  # Soft character limit

    def __init__(self):
        self.enhancement_history: List[str] = []

    def validate(self, prompt: str) -> Tuple[bool, List[str]]:
        """Validate prompt and return (is_valid, warnings)."""
        warnings = []

        if not prompt or not prompt.strip():
            return False, ["Prompt cannot be empty"]

        if len(prompt) > 1000:
            warnings.append(f"Prompt is very long ({len(prompt)} chars). Consider shortening.")

        estimated_tokens = len(prompt.split())
        if estimated_tokens > self.MAX_TOKENS:
            warnings.append(
                f"Prompt may exceed CLIP's {self.MAX_TOKENS} token limit "
                f"(~{estimated_tokens} words). Tokens beyond limit will be ignored."
            )

        if len(prompt) < 5:
            warnings.append("Very short prompt - consider adding more descriptive details")

        # Check for potential issues
        if prompt.lower() == prompt and len(prompt) > 50:
            warnings.append("Tip: Capitalizing key concepts can help emphasize them")

        return True, warnings

    def clean(self, prompt: str) -> str:
        """Clean and normalize prompt text."""
        # Remove excessive whitespace
        prompt = re.sub(r'\s+', ' ', prompt).strip()
        # Remove duplicate commas
        prompt = re.sub(r',\s*,+', ',', prompt)
        # Remove trailing/leading commas
        prompt = prompt.strip(',').strip()
        return prompt

    def enhance(
        self,
        prompt: str,
        style_preset: str = "None",
        quality_boost: bool = False,
    ) -> str:
        """
        Enhance prompt with style and quality modifiers.

        Args:
            prompt: Base prompt text
            style_preset: Style preset name from STYLE_PRESETS
            quality_boost: Add quality booster tokens

        Returns:
            Enhanced prompt string
        """
        parts = [self.clean(prompt)]

        if style_preset and style_preset != "None":
            style_modifier = STYLE_PRESETS.get(style_preset, "")
            if style_modifier:
                parts.append(style_modifier)

        if quality_boost:
            parts.append(", ".join(QUALITY_BOOSTERS[:4]))  # Top 4 boosters

        enhanced = ", ".join(filter(None, parts))
        return self.clean(enhanced)

    def build_negative_prompt(
        self,
        negative_preset: str = "Default",
        custom_negative: str = "",
    ) -> str:
        """Build final negative prompt from preset + custom."""
        parts = []

        preset_neg = NEGATIVE_PRESETS.get(negative_preset, "")
        if preset_neg:
            parts.append(preset_neg)

        if custom_negative and custom_negative.strip():
            parts.append(self.clean(custom_negative))

        return ", ".join(filter(None, parts))

    def process(
        self,
        prompt: str,
        negative_prompt: str = "",
        style_preset: str = "None",
        negative_preset: str = "Default",
        quality_boost: bool = False,
    ) -> ProcessedPrompt:
        """
        Full prompt processing pipeline.

        Returns:
            ProcessedPrompt with all fields populated
        """
        is_valid, warnings = self.validate(prompt)
        if not is_valid:
            raise ValueError(f"Invalid prompt: {warnings[0]}")

        enhanced = self.enhance(prompt, style_preset, quality_boost)
        final_negative = self.build_negative_prompt(negative_preset, negative_prompt)

        token_count = len(enhanced.split())

        result = ProcessedPrompt(
            original=prompt,
            enhanced=enhanced,
            negative=final_negative,
            style_applied=style_preset,
            quality_boost=quality_boost,
            token_count=token_count,
            warnings=warnings,
        )

        logger.debug(f"Processed prompt: '{enhanced[:80]}...' ({token_count} tokens)")
        return result

    @staticmethod
    def get_style_presets() -> List[str]:
        return list(STYLE_PRESETS.keys())

    @staticmethod
    def get_negative_presets() -> List[str]:
        return list(NEGATIVE_PRESETS.keys())

    @staticmethod
    def get_example_prompts() -> List[str]:
        return [
            "A majestic dragon soaring over a misty mountain range at sunset",
            "Portrait of a cyberpunk woman with neon lights reflecting in her eyes",
            "Ancient Japanese temple surrounded by cherry blossoms in spring",
            "An astronaut exploring an alien planet with two moons",
            "Cozy cottage in an enchanted forest with glowing fireflies",
            "Epic fantasy battle scene with knights and magical creatures",
            "Serene underwater scene with bioluminescent sea creatures",
            "Futuristic city skyline at night with flying cars",
        ]
