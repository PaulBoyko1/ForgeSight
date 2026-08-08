# ForgeSight

**Industrial visual anomaly detection that treats distribution shift as a first-class failure mode.**

ForgeSight is an end-to-end computer-vision research and deployment project for unsupervised industrial inspection. It trains only on defect-free images, builds a compact memory bank of normal patch embeddings, localizes anomalous regions by nearest-neighbor distance, calibrates a decision threshold on normal validation data, and reports both predictive quality and deployment cost.

The first built-in model is **PatchMemory**, a transparent PatchCore-style baseline implemented in this repository rather than delegated to a large anomaly-detection framework. It is intentionally simple enough to audit and extend. The benchmark target is MVTec AD 2, whose public test data includes acquisition/lighting changes that make robustness under distribution shift measurable.

## What is implemented

- MVTec-style / MVTec AD 2 dataset discovery into an explicit sample manifest
- frozen ResNet-18 multi-scale patch feature extractor
- deterministic farthest-first coreset selection
- chunked nearest-neighbor anomaly scoring
- image-level scores and pixel-resolution anomaly maps
- validation-only quantile threshold calibration
- image AUROC / average precision and pixel AUROC
- domain/shift-group evaluation helper
- inference latency, throughput, model-memory benchmarks
- portable checkpoints containing model state + calibrated threshold
- FastAPI image inference endpoint
- CPU-safe unit tests and GitHub Actions

## Why MVTec AD 2

Classic anomaly-detection benchmarks became increasingly saturated. MVTec AD 2 adds eight industrial scenarios with more than 8,000 high-resolution images and test conditions whose lighting is not necessarily represented in training data. ForgeSight is designed around the question: **how much anomaly-detection performance survives when acquisition conditions change?**

The dataset itself is not included in this repository and has its own non-commercial license.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e '.[dev]'
pytest
```

Create a manifest for one MVTec AD 2 category:

```bash
forgesight manifest /data/mvtec_ad_2/can --output artifacts/can_manifest.csv
```

Train a normal-only PatchMemory model:

```bash
forgesight fit /data/mvtec_ad_2/can \
  --checkpoint artifacts/can_patchmemory.pt \
  --image-size 256 \
  --coreset-ratio 0.05
```

The first run with pretrained weights may download torchvision's ResNet-18 weights.

## Architecture

```text
normal training images
        |
        v
frozen multi-scale CNN features
        |
        v
patch embeddings ----> deterministic coreset ----> normal memory bank
                                                \
validation-normal images --------------------------> threshold calibration

new inspection image
        |
        v
patch embeddings --> nearest-normal-patch distance --> anomaly heatmap
                                      |
                                      +--> image anomaly score
```

## Research roadmap

- [x] auditable PatchCore-style baseline
- [x] threshold calibration separated from test evaluation
- [x] image + pixel metrics
- [x] latency/memory benchmark hooks
- [x] checkpoint + API inference path
- [ ] run full public MVTec AD 2 benchmark
- [ ] report performance by acquisition/lighting domain
- [ ] compare against EfficientAD through a controlled adapter
- [ ] add a foundation-feature baseline (DINO-family) under identical evaluation
- [ ] calibration curves and threshold-transfer experiments
- [ ] ONNX/OpenVINO export for deployable backbones where supported
- [ ] failure-case gallery and browser demo

## Design rules

1. Training data is normal-only unless an experiment explicitly says otherwise.
2. Test labels never select thresholds or hyperparameters.
3. Accuracy and systems metrics are reported together: AUROC without latency/memory is incomplete for an inspection system.
4. Dataset license restrictions are kept separate from this repository's code license.
5. A baseline is labeled honestly; PatchMemory is PatchCore-inspired, not claimed as an exact paper reproduction.

## License

ForgeSight code is MIT licensed. Datasets and pretrained weights retain their own licenses.
