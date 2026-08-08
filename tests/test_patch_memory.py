import torch
from torch import nn

from forgesight.models.patch_memory import PatchMemory, farthest_first_coreset


class ToyExtractor(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images[:, :2, ::2, ::2]


def test_coreset_is_deterministic_and_reduces_features() -> None:
    features = torch.arange(80, dtype=torch.float32).reshape(20, 4)
    a = farthest_first_coreset(features, ratio=0.25, seed=7)
    b = farthest_first_coreset(features, ratio=0.25, seed=7)
    assert a.shape == (5, 4)
    assert torch.equal(a, b)


def test_patch_memory_fit_predict_and_threshold(tmp_path) -> None:
    model = PatchMemory(ToyExtractor(), coreset_ratio=0.5)
    normal = torch.zeros(2, 3, 8, 8)
    model.fit_embeddings(normal)
    prediction = model.predict(torch.ones(1, 3, 8, 8))
    assert prediction.image_scores.shape == (1,)
    assert prediction.anomaly_maps.shape == (1, 1, 8, 8)
    assert prediction.image_scores.item() > 0
    threshold = model.calibrate_threshold(torch.tensor([0.1, 0.2, 0.3]), quantile=0.9)
    assert 0.2 < threshold <= 0.3


def test_resnet_extractor_checkpoint_roundtrip(tmp_path) -> None:
    from forgesight.models.features import ResNet18PatchExtractor

    extractor = ResNet18PatchExtractor(pretrained=False)
    assert all(not parameter.requires_grad for parameter in extractor.parameters())
    model = PatchMemory(extractor, coreset_ratio=0.1)
    normal = torch.zeros(1, 3, 32, 32)
    model.fit_embeddings(normal)
    path = tmp_path / "model.pt"
    model.save(path)
    restored = PatchMemory.load(path)
    prediction = restored.predict(normal)
    assert prediction.anomaly_maps.shape == (1, 1, 32, 32)


def test_benchmark_reports_system_metrics() -> None:
    from forgesight.evaluation.benchmark import benchmark_inference

    model = PatchMemory(ToyExtractor(), coreset_ratio=1.0)
    sample = torch.zeros(1, 3, 8, 8)
    model.fit_embeddings(sample)
    result = benchmark_inference(model, sample, warmup=0, iterations=2)
    assert result["latency_median_ms"] > 0
    assert result["images_per_second"] > 0
    assert result["memory_bank_mb"] > 0
