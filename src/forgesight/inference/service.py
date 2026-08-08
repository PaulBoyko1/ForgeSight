"""FastAPI deployment surface for a fitted ForgeSight checkpoint."""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision.transforms import v2

from forgesight.models.patch_memory import PatchMemory


def build_preprocess(image_size: int = 256) -> v2.Compose:
    return v2.Compose(
        [
            v2.Resize((image_size, image_size), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def create_app(checkpoint: str | Path, *, image_size: int = 256) -> Any:
    try:
        from fastapi import FastAPI, File, HTTPException, UploadFile
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install ForgeSight with the 'serve' extra") from exc

    model = PatchMemory.load(checkpoint)
    preprocess = build_preprocess(image_size)
    app = FastAPI(title="ForgeSight", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)) -> dict[str, float | bool | None]:  # noqa: B008
        try:
            payload = await file.read()
            image = Image.open(io.BytesIO(payload)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail="invalid image") from exc
        tensor = preprocess(image)[None, ...]
        prediction = model.predict(tensor)
        score = float(prediction.image_scores.item())
        threshold = model.threshold
        return {
            "score": score,
            "threshold": threshold,
            "anomalous": None if threshold is None else score > threshold,
        }

    return app


def app_from_env() -> Any:
    checkpoint = os.environ.get("FORGESIGHT_CHECKPOINT")
    if not checkpoint:
        raise RuntimeError("FORGESIGHT_CHECKPOINT is required")
    image_size = int(os.environ.get("FORGESIGHT_IMAGE_SIZE", "256"))
    return create_app(checkpoint, image_size=image_size)
