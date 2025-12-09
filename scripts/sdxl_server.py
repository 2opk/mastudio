#!/usr/bin/env python3
"""
Minimal SDXL HTTP server to avoid reloading the pipeline per run.

Usage:
  CUDA_VISIBLE_DEVICES=0 python -m scripts.sdxl_server --port 7000 --device cuda:0

Endpoint:
  POST /generate {"prompts": ["..."]} -> {"images": ["<base64 png>", ...]}
"""

import argparse
import base64
import logging
import threading
from io import BytesIO
from typing import List

import torch
import uvicorn
from diffusers import DiffusionPipeline
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI()
pipe = None
_lock = threading.Lock()
logger = logging.getLogger("sdxl_server")
logging.basicConfig(level=logging.INFO)


class GenerateRequest(BaseModel):
    prompts: List[str]


class GenerateResponse(BaseModel):
    images: List[str]


def load_pipeline(device: str) -> None:
    global pipe
    if pipe is not None:
        return
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    pipe.to(device)


def _reset_scheduler() -> None:
    """Reset scheduler state to avoid step_index drift."""
    global pipe
    if pipe is None:
        return
    try:
        cls = pipe.scheduler.__class__
        cfg = pipe.scheduler.config
        pipe.scheduler = cls.from_config(cfg)
        if hasattr(pipe.scheduler, "step_index"):
            pipe.scheduler.step_index = None
    except Exception:
        logger.exception("Failed to reset scheduler")


def _run_pipe(prompts: List[str]):
    if pipe is None:
        raise RuntimeError("Pipeline not loaded")
    return pipe(prompt=prompts).images


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    if pipe is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded yet")
    if not req.prompts:
        raise HTTPException(status_code=400, detail="prompts required")
    try:
        with _lock:
            images = _run_pipe(req.prompts)
        encoded: List[str] = []
        for img in images:
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            encoded.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))
        return GenerateResponse(images=encoded)
    except Exception as exc:
        logger.exception("Generation failed; resetting scheduler and retrying once")
        try:
            with _lock:
                _reset_scheduler()
                images = _run_pipe(req.prompts)
            encoded: List[str] = []
            for img in images:
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                encoded.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))
            return GenerateResponse(images=encoded)
        except Exception as exc2:
            logger.exception("Retry after scheduler reset failed")
            raise HTTPException(status_code=500, detail=str(exc2))


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "loaded": pipe is not None}


def main():
    parser = argparse.ArgumentParser(description="Run a lightweight SDXL HTTP server.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=7000, help="Bind port")
    parser.add_argument("--device", default="cuda:0", help="Torch device for SDXL")
    args = parser.parse_args()

    load_pipeline(args.device)
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
