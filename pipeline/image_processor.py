# pipeline/image_processor.py
"""
Image Processor: Post-processing, metadata embedding, and storage management.
"""

import io
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

from PIL import Image, PngImagePlugin
from loguru import logger


class ImageProcessor:
    """
    Post-processing pipeline for generated images.
    Handles metadata, saving, and basic image operations.
    """

    def __init__(self, output_dir: str = "./artifacts/generated"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def embed_metadata(
        self,
        image: Image.Image,
        metadata: Dict[str, Any],
    ) -> Image.Image:
        """
        Embed generation metadata into PNG image as text chunks.
        Allows recovering parameters from image files.
        """
        if image.format != "PNG" and image.mode in ("RGB", "RGBA"):
            png_info = PngImagePlugin.PngInfo()
            for key, value in metadata.items():
                png_info.add_text(str(key), str(value))
            # Convert to bytes and reload to attach metadata
            buffer = io.BytesIO()
            image.save(buffer, format="PNG", pnginfo=png_info)
            buffer.seek(0)
            return Image.open(buffer)
        return image

    def save_image(
        self,
        image: Image.Image,
        request_id: str,
        index: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
        save_metadata_json: bool = True,
    ) -> Path:
        """
        Save image and optionally its metadata JSON sidecar.

        Returns:
            Path to saved image
        """
        timestamp = int(time.time())
        filename = f"{timestamp}_{request_id}_{index}.png"
        filepath = self.output_dir / filename

        if metadata:
            image = self.embed_metadata(image, metadata)
            if save_metadata_json:
                json_path = self.output_dir / f"{timestamp}_{request_id}_{index}.json"
                with open(json_path, "w") as f:
                    json.dump(metadata, f, indent=2, default=str)

        image.save(filepath, format="PNG", optimize=False)
        logger.info(f"Saved image: {filepath}")
        return filepath

    def image_to_bytes(self, image: Image.Image, format: str = "PNG") -> bytes:
        """Convert PIL Image to bytes for in-memory use."""
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        return buffer.getvalue()

    def resize_for_display(
        self,
        image: Image.Image,
        max_size: int = 768,
    ) -> Image.Image:
        """Resize image for display if larger than max_size."""
        w, h = image.size
        if max(w, h) <= max_size:
            return image
        scale = max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        return image.resize((new_w, new_h), Image.LANCZOS)

    def create_grid(
        self,
        images: List[Image.Image],
        cols: int = 2,
        padding: int = 8,
        background_color: tuple = (30, 30, 30),
    ) -> Image.Image:
        """
        Arrange multiple images into a grid layout.

        Args:
            images: List of PIL Images (must all be same size)
            cols: Number of columns
            padding: Pixel padding between images
            background_color: Background color RGB tuple

        Returns:
            Grid image
        """
        if not images:
            raise ValueError("No images provided")
        if len(images) == 1:
            return images[0]

        rows = (len(images) + cols - 1) // cols
        w, h = images[0].size

        grid_w = cols * w + (cols + 1) * padding
        grid_h = rows * h + (rows + 1) * padding
        grid = Image.new("RGB", (grid_w, grid_h), background_color)

        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols
            x = padding + col * (w + padding)
            y = padding + row * (h + padding)
            if img.size != (w, h):
                img = img.resize((w, h), Image.LANCZOS)
            grid.paste(img, (x, y))

        return grid

    def get_stored_images(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List recently generated images with metadata."""
        images = []
        for json_file in sorted(
            self.output_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:limit]:
            try:
                with open(json_file) as f:
                    meta = json.load(f)
                img_file = json_file.with_suffix(".png")
                if img_file.exists():
                    images.append({"path": str(img_file), "metadata": meta})
            except Exception:
                pass
        return images

    def cleanup_old_images(self, keep_last: int = 200):
        """Remove old generated images, keeping the most recent N."""
        all_images = sorted(
            self.output_dir.glob("*.png"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for old_img in all_images[keep_last:]:
            old_img.unlink(missing_ok=True)
            json_sidecar = old_img.with_suffix(".json")
            json_sidecar.unlink(missing_ok=True)
            logger.debug(f"Cleaned up: {old_img.name}")
