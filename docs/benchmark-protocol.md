# Benchmark protocol

ForgeSight treats anomaly accuracy and deployment cost as one experiment, not two unrelated demos.

## Dataset

Primary target: MVTec AD 2 public evaluation data. The repository does not redistribute the dataset. Training and validation images are normal-only; the public test split is used for local labeled evaluation. Private test data is reserved for the official evaluation path.

## Model comparison contract

Each baseline must use the same category split, image resolution, public test labels, and reporting code. Thresholds are calibrated only from the validation split. Test labels cannot influence thresholds or hyperparameters.

Planned comparison:

1. PatchMemory (repository-native PatchCore-style baseline)
2. EfficientAD adapter
3. foundation-feature nearest-neighbor baseline

## Predictive metrics

- image AUROC
- image average precision
- pixel AUROC when public masks are available
- thresholded precision/recall/F1 after validation-only calibration
- results per category and acquisition domain where domain metadata is available
- degradation relative to a designated reference acquisition domain

## Systems metrics

Measure on CPU and available GPU separately:

- median latency
- p95 latency
- images/second
- peak device memory when available
- checkpoint size
- memory-bank/model resident size

Warm-up iterations are excluded from latency statistics. Hardware, precision, batch size, image size, and software versions must be recorded with results.

## Failure analysis

Every benchmark report should include representative false positives and false negatives, especially failures correlated with lighting or acquisition changes. Negative results remain in the repository.
