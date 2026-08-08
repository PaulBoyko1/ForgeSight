"""ForgeSight command-line workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from forgesight.data.mvtec import ManifestDataset, discover_category
from forgesight.evaluation.benchmark import benchmark_inference
from forgesight.evaluation.metrics import binary_anomaly_metrics, pixel_auroc
from forgesight.evaluation.shift import metrics_by_domain
from forgesight.models.patch_memory import PatchMemory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forgesight")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest", help="discover a MVTec-style category")
    manifest.add_argument("category", type=Path)
    manifest.add_argument("--output", type=Path, required=True)

    fit = subparsers.add_parser("fit", help="fit PatchMemory on normal training images")
    fit.add_argument("category", type=Path)
    fit.add_argument("--checkpoint", type=Path, required=True)
    fit.add_argument("--image-size", type=int, default=256)
    fit.add_argument("--batch-size", type=int, default=8)
    fit.add_argument("--coreset-ratio", type=float, default=0.05)
    fit.add_argument("--validation-quantile", type=float, default=0.995)

    evaluate = subparsers.add_parser("evaluate", help="evaluate a checkpoint on public test data")
    evaluate.add_argument("category", type=Path)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--image-size", type=int, default=256)
    evaluate.add_argument("--batch-size", type=int, default=8)

    benchmark = subparsers.add_parser("benchmark", help="benchmark checkpoint inference")
    benchmark.add_argument("--checkpoint", type=Path, required=True)
    benchmark.add_argument("--image-size", type=int, default=256)
    benchmark.add_argument("--batch-size", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "benchmark":
        model = PatchMemory.load(args.checkpoint)
        sample = torch.zeros(args.batch_size, 3, args.image_size, args.image_size)
        print(json.dumps(benchmark_inference(model, sample), indent=2))
        return

    manifest = discover_category(args.category)
    if args.command == "manifest":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(args.output, index=False)
        print(f"wrote {len(manifest)} samples to {args.output}")
        return
    if args.command == "fit":
        _fit(args, manifest)
        return
    _evaluate(args, manifest)


def _fit(args: argparse.Namespace, manifest: pd.DataFrame) -> None:
    train = manifest[(manifest["split"] == "train") & (manifest["label"] == 0)]
    validation = manifest[(manifest["split"] == "validation") & (manifest["label"] == 0)]
    if train.empty:
        raise SystemExit("no normal training images found")
    train_loader = DataLoader(
        ManifestDataset(train, image_size=args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
    )
    model = PatchMemory(coreset_ratio=args.coreset_ratio)
    model.fit_loader(train_loader)

    if not validation.empty:
        validation_loader = DataLoader(
            ManifestDataset(validation, image_size=args.image_size),
            batch_size=args.batch_size,
            shuffle=False,
        )
        scores: list[torch.Tensor] = []
        for images, _, _, _ in validation_loader:
            scores.append(model.predict(images).image_scores.cpu())
        model.calibrate_threshold(torch.cat(scores), quantile=args.validation_quantile)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.checkpoint)
    print(f"saved checkpoint to {args.checkpoint}")
    if model.threshold is not None:
        print(f"validation threshold: {model.threshold:.6f}")


def _evaluate(args: argparse.Namespace, manifest: pd.DataFrame) -> None:
    test = manifest[manifest["split"].isin(["test", "test_public"])].reset_index(drop=True)
    if test.empty:
        raise SystemExit("no labeled public test images found")
    model = PatchMemory.load(args.checkpoint)
    loader = DataLoader(
        ManifestDataset(test, image_size=args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
    )
    scores: list[np.ndarray] = []
    maps: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for images, _, batch_masks, _ in loader:
        prediction = model.predict(images)
        scores.append(prediction.image_scores.cpu().numpy())
        maps.append(prediction.anomaly_maps.cpu().numpy())
        masks.append(batch_masks.numpy())
    all_scores = np.concatenate(scores)
    labels = test["label"].to_numpy(dtype=int)
    report: dict[str, object] = {"image": binary_anomaly_metrics(labels, all_scores)}
    all_masks = np.concatenate(masks)
    all_maps = np.concatenate(maps)
    anomalous_rows = test["label"].to_numpy(dtype=bool)
    if not anomalous_rows.any() or test.loc[anomalous_rows, "mask_path"].notna().all():
        if len(np.unique(all_masks)) >= 2:
            report["pixel_auroc"] = pixel_auroc(all_masks, all_maps)
    domain_frame = test[["domain", "label"]].copy()
    domain_frame["score"] = all_scores
    grouped = metrics_by_domain(domain_frame)
    report["domains"] = grouped.to_dict(orient="records")
    print(json.dumps(report, indent=2))
