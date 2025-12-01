import base64
import json
import struct
import uuid
import zlib
from pathlib import Path
from typing import Dict, List, Optional

import httpx

from .utils import SDXLConfig, ensure_output_dir


class SDXLWrapper:
    def __init__(self, config: SDXLConfig, stability_key: Optional[str] = None) -> None:
        self.config = config
        self.stability_key = stability_key
        self.output_dir = ensure_output_dir(config.output_dir)

    def generate(self, prompts: List[str]) -> List[str]:
        if self.config.mode == "stability" and self.stability_key:
            return self._generate_via_api(prompts)
        return self._generate_mock(prompts)

    def _generate_via_api(self, prompts: List[str]) -> List[str]:
        images: List[str] = []
        url = "https://api.stability.ai/v1/generation/sdxl-1024x1024/text-to-image"
        headers = {
            "Authorization": f"Bearer {self.stability_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        for prompt in prompts:
            payload = {
                "text_prompts": [{"text": prompt, "weight": 1}],
                "cfg_scale": 7,
                "clip_guidance_preset": "FAST_BLUE",
                "samples": 1,
                "steps": 30,
            }
            response = httpx.post(url, headers=headers, json=payload, timeout=180)
            response.raise_for_status()
            data = response.json()
            artifact = data["artifacts"][0]
            image_bytes = base64.b64decode(artifact["base64"])
            filename = self._write_image_bytes(image_bytes, suffix=".png")
            images.append(str(filename))
        return images

    def _generate_mock(self, prompts: List[str]) -> List[str]:
        """Fallback when API key is missing. Writes simple placeholder PNGs and prompt metadata."""
        placeholder_png = self._solid_png_bytes(128, 128, (210, 210, 210, 255))
        images: List[str] = []
        for prompt in prompts:
            filename = self._write_image_bytes(placeholder_png, suffix=".png")
            metadata_path = filename.with_suffix(".txt")
            metadata_path.write_text(f"MOCK IMAGE FOR PROMPT:\n{prompt}\n", encoding="utf-8")
            images.append(str(filename))
        return images

    def _solid_png_bytes(self, width: int, height: int, rgba: tuple[int, int, int, int]) -> bytes:
        """Create a solid-color PNG without external deps."""
        r, g, b, a = rgba
        # Build raw RGBA data with filter byte per scanline.
        row = bytes([0] + [r, g, b, a] * width)
        raw = row * height
        compressed = zlib.compress(raw, level=9)

        def _chunk(chunk_type: bytes, data: bytes) -> bytes:
            return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", zlib.crc32(chunk_type + data))

        ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)  # 8-bit RGBA
        png = b"\x89PNG\r\n\x1a\n"
        png += _chunk(b"IHDR", ihdr)
        png += _chunk(b"IDAT", compressed)
        png += _chunk(b"IEND", b"")
        return png

    def _write_image_bytes(self, data: bytes, suffix: str) -> Path:
        name = f"image_{uuid.uuid4().hex}{suffix}"
        path = self.output_dir / name
        path.write_bytes(data)
        return path


def save_json_report(report: Dict, path: Path) -> None:
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
