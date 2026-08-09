import io

import torch
from fastapi.testclient import TestClient
from PIL import Image

from forgesight.inference.service import create_app
from forgesight.models.patch_memory import Prediction


class StubModel:
    threshold = 0.5

    def eval(self) -> StubModel:
        return self

    def predict(self, images: torch.Tensor) -> Prediction:
        return Prediction(
            image_scores=torch.tensor([0.75]),
            anomaly_maps=torch.zeros((1, 1, *images.shape[-2:])),
        )


def image_payload(size: tuple[int, int] = (4, 4)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color="white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_inference_service_enforces_upload_contracts(monkeypatch) -> None:
    monkeypatch.setattr(
        "forgesight.inference.service.PatchMemory.load",
        lambda _: StubModel(),
    )
    app = create_app(
        "checkpoint.pt",
        image_size=32,
        max_upload_bytes=1024,
        max_image_pixels=64,
    )

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200

        valid = client.post(
            "/predict",
            files={"file": ("sample.png", image_payload(), "image/png")},
        )
        assert valid.status_code == 200
        assert valid.json()["anomalous"] is True

        unsupported = client.post(
            "/predict",
            files={"file": ("sample.txt", b"payload", "text/plain")},
        )
        assert unsupported.status_code == 415

        oversized = client.post(
            "/predict",
            files={"file": ("sample.png", b"x" * 1025, "image/png")},
        )
        assert oversized.status_code == 413

        oversized_image = client.post(
            "/predict",
            files={"file": ("large.png", image_payload((9, 9)), "image/png")},
        )
        assert oversized_image.status_code == 400
