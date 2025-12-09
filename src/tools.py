import base64
import json
import os
import struct
import uuid
import zlib
from io import BytesIO
from pathlib import Path
from typing import Callable, Dict, List, Optional

import requests
import torch
from diffusers import DiffusionPipeline

from .utils import SDXLConfig, ensure_output_dir


class SDXLWrapper:
    def __init__(self, config: SDXLConfig) -> None:
        self.config = config
        self.output_dir = ensure_output_dir(config.output_dir)
        self.pipe = None
        # Skip local load if an external server is configured.
        if self.config.mode == "local" and not (self.config.api_base):
            self.load_model()

    def load_model(self):
        if self.pipe is not None:
            return
        self.pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        self.pipe.to(self.config.device)

    def generate(self, prompts: List[str], iteration: int) -> List[str]:
        images: List[str] = []
        mode = "server" if self.config.api_base else self.config.mode
        if mode == "local":
            if self.pipe is None:
                self.load_model()
            if len(prompts) > 1:
                batch_bytes = self._generate_sdxl_batch(prompts)
                for idx, (prompt, image_bytes) in enumerate(zip(prompts, batch_bytes)):
                    filename = self.get_image_path(iteration=iteration, prompt_index=idx, suffix=".png")
                    filename.write_bytes(image_bytes)
                    images.append(str(filename))
                return images
            generator: Callable[[str], bytes] = self._generate_sdxl
        elif mode == "server":
            batch_bytes: Optional[List[bytes]] = None
            try:
                batch_bytes = self._generate_via_server_batch(prompts)
            except Exception as exc:
                if self.config.failover_mode == "mock":
                    print(f"[SDXL] server failed ({exc}); falling back to mock.")
                    generator = self._generate_mock
                else:
                    raise
            if batch_bytes is not None:
                for idx, image_bytes in enumerate(batch_bytes):
                    filename = self.get_image_path(iteration=iteration, prompt_index=idx, suffix=".png")
                    filename.write_bytes(image_bytes)
                    images.append(str(filename))
                return images
        elif mode == "api":
            generator = self._generate_via_api
        else:
            generator = self._generate_mock

        for idx, prompt in enumerate(prompts):
            image_bytes = generator(prompt)
            filename = self.get_image_path(iteration=iteration, prompt_index=idx, suffix=".png")
            filename.write_bytes(image_bytes)
            if generator is self._generate_mock:
                metadata_path = filename.with_suffix(".txt")
                metadata_path.write_text(f"MOCK IMAGE FOR PROMPT:\n{prompt}\n", encoding="utf-8")
            images.append(str(filename))
        return images

    def _generate_sdxl(self, prompt: str) -> bytes:
        image = self.pipe(prompt=prompt).images[0]
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def _generate_sdxl_batch(self, prompts: List[str]) -> List[bytes]:
        result = self.pipe(prompt=prompts).images
        outputs: List[bytes] = []
        for img in result:
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            outputs.append(buffer.getvalue())
        return outputs

    def _generate_via_api(self, prompt: str) -> bytes:
        stability_key = os.getenv("STABILITY_API_KEY", "")
        if not stability_key:
            raise RuntimeError("STABILITY_API_KEY is required for SDXL mode 'api'")
        url = "https://api.stability.ai/v2beta/stable-image/generate/core"
        headers = {
            "authorization": f"Bearer {stability_key}",
            "accept": "image/*"
        }
        files = { "none": "" }

        data = {
            "prompt": prompt,
            "output_format": "png"
        }
        response = requests.post(url, headers=headers, files=files, data=data)
        return response.content

    def _generate_via_server_batch(self, prompts: List[str]) -> List[bytes]:
        base = (self.config.api_base or "").rstrip("/")
        if not base:
            raise RuntimeError("SDXL api_base is required for HTTP server mode (set SDXL_API_BASE or config.sdxl.api_base)")
        url = f"{base}/generate"
        resp = requests.post(url, json={"prompts": prompts}, timeout=300)
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            detail = ""
            try:
                detail = f" body={resp.text}"
            except Exception:
                detail = ""
            raise RuntimeError(f"SDXL server error {resp.status_code}{detail}") from exc
        data = resp.json()
        images_b64 = data.get("images") or []
        if len(images_b64) != len(prompts):
            raise RuntimeError("SDXL server returned unexpected number of images")
        decoded: List[bytes] = []
        for item in images_b64:
            try:
                decoded.append(base64.b64decode(item))
            except Exception as exc:
                raise RuntimeError("Failed to decode SDXL server image payload") from exc
        return decoded

    def _generate_mock(self, prompt: str) -> bytes:
        """Fallback when API key is missing. Writes simple placeholder PNGs and prompt metadata."""
        return self._solid_png_bytes(128, 128, (210, 210, 210, 255))

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

    def get_image_path(self, *, iteration: int, prompt_index: int, suffix: str) -> Path:
        name = f"r{iteration}_p{prompt_index+1}_{uuid.uuid4().hex}{suffix}"
        path = self.output_dir / name
        return path


def save_json_report(report: Dict, path: Path) -> None:
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
