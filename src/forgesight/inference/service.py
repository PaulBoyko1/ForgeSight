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

ALLOWED_IMAGE_TYPES = frozenset({"image/bmp", "image/jpeg", "image/png", "image/webp"})
DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024
DEFAULT_MAX_IMAGE_PIXELS = 20_000_000


def build_preprocess(image_size: int = 256) -> v2.Compose:
    return v2.Compose(
        [
            v2.Resize((image_size, image_size), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _positive_int(value: int, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _decode_image(payload: bytes, *, max_image_pixels: int) -> Image.Image:
    try:
        with Image.open(io.BytesIO(payload)) as source:
            if source.width * source.height > max_image_pixels:
                raise ValueError("image dimensions exceed the configured limit")
            return source.convert("RGB")
    except (Image.DecompressionBombError, Image.UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("invalid or oversized image") from exc


def create_app(
    checkpoint: str | Path,
    *,
    image_size: int = 256,
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
    max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
) -> Any:
    try:
        from fastapi import FastAPI, File, HTTPException, UploadFile
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install ForgeSight with the 'serve' extra") from exc

    resolved_image_size = _positive_int(image_size, name="image_size", minimum=32)
    upload_limit = _positive_int(max_upload_bytes, name="max_upload_bytes", minimum=1)
    pixel_limit = _positive_int(max_image_pixels, name="max_image_pixels", minimum=1)
    model = PatchMemory.load(checkpoint)
    model.eval()
    preprocess = build_preprocess(resolved_image_size)
    app = FastAPI(title="ForgeSight", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)) -> dict[str, float | bool | None]:  # noqa: B008
        if file.content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(status_code=415, detail="unsupported image content type")
        try:
            payload = await file.read(upload_limit + 1)
        finally:
            await file.close()
        if len(payload) > upload_limit:
            raise HTTPException(status_code=413, detail="uploaded image is too large")
        try:
            image = _decode_image(payload, max_image_pixels=pixel_limit)
        except ValueError as exc:
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


def _env_int(name: str, default: int, *, minimum: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    try:
        return _positive_int(parsed, name=name, minimum=minimum)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


def app_from_env() -> Any:
    checkpoint = os.environ.get("FORGESIGHT_CHECKPOINT")
    if not checkpoint:
        raise RuntimeError("FORGESIGHT_CHECKPOINT is required")
    return create_app(
        checkpoint,
        image_size=_env_int("FORGESIGHT_IMAGE_SIZE", 256, minimum=32),
        max_upload_bytes=_env_int(
            "FORGESIGHT_MAX_UPLOAD_BYTES",
            DEFAULT_MAX_UPLOAD_BYTES,
            minimum=1,
        ),
        max_image_pixels=_env_int(
            "FORGESIGHT_MAX_IMAGE_PIXELS",
            DEFAULT_MAX_IMAGE_PIXELS,
            minimum=1,
        ),
    )
