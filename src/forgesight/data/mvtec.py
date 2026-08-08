"""MVTec-style dataset discovery and image loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
NORMAL_TOKENS = {"good", "normal", "ok"}
PUBLIC_SPLITS = ("train", "validation", "test", "test_public")


@dataclass(frozen=True)
class SampleRecord:
    image_path: str
    split: str
    label: int
    defect_type: str
    domain: str
    mask_path: str | None = None


def discover_category(category_root: str | Path) -> pd.DataFrame:
    """Discover a MVTec-style category into an explicit manifest.

    The scanner supports classic ``train/test/ground_truth`` layouts and AD 2's
    ``train/validation/test_public`` split names. Unknown private splits are not
    assigned labels and are therefore intentionally omitted from this local-eval manifest.
    """
    root = Path(category_root)
    if not root.is_dir():
        raise ValueError(f"category root does not exist: {root}")

    records: list[SampleRecord] = []
    for split in PUBLIC_SPLITS:
        split_root = root / split
        if not split_root.is_dir():
            continue
        for image_path in sorted(p for p in split_root.rglob("*") if _is_image(p)):
            rel = image_path.relative_to(split_root)
            defect_type = rel.parts[0] if len(rel.parts) > 1 else "good"
            label = (
                0
                if split in {"train", "validation"}
                else int(defect_type.lower() not in NORMAL_TOKENS)
            )
            domain = _infer_domain(rel, defect_type)
            mask = _find_mask(root, split, image_path, defect_type) if label == 1 else None
            records.append(
                SampleRecord(
                    image_path=str(image_path),
                    split=split,
                    label=label,
                    defect_type=defect_type,
                    domain=domain,
                    mask_path=str(mask) if mask is not None else None,
                )
            )
    if not records:
        raise ValueError(f"no public MVTec-style images found under {root}")
    return pd.DataFrame([asdict(record) for record in records])


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def _infer_domain(relative_path: Path, defect_type: str) -> str:
    # Some datasets encode acquisition/lighting conditions as intermediate folders.
    parents = relative_path.parts[:-1]
    candidates = [part for part in parents if part.lower() != defect_type.lower()]
    return "/".join(candidates) if candidates else "default"


def _find_mask(root: Path, split: str, image_path: Path, defect_type: str) -> Path | None:
    candidates = [root / "ground_truth", root / f"ground_truth_{split}", root / "masks"]
    stem_variants = (f"{image_path.stem}_mask", image_path.stem)
    for base in candidates:
        defect_root = base / defect_type
        if not defect_root.exists():
            continue
        for stem in stem_variants:
            for suffix in IMAGE_SUFFIXES:
                candidate = defect_root / f"{stem}{suffix}"
                if candidate.exists():
                    return candidate
        matches = list(defect_root.rglob(f"{image_path.stem}*"))
        image_matches = [match for match in matches if _is_image(match)]
        if len(image_matches) == 1:
            return image_matches[0]
    return None


class ManifestDataset(Dataset[tuple[torch.Tensor, int, torch.Tensor, str]]):
    """Load RGB images and optional binary masks from a manifest DataFrame."""

    def __init__(self, manifest: pd.DataFrame, *, image_size: int = 256) -> None:
        required = {"image_path", "label", "mask_path", "domain"}
        missing = required - set(manifest.columns)
        if missing:
            raise ValueError(f"manifest missing columns: {sorted(missing)}")
        if image_size < 32:
            raise ValueError("image_size must be >= 32")
        self.manifest = manifest.reset_index(drop=True).copy()
        self.image_size = image_size
        self.transform = v2.Compose(
            [
                v2.Resize((image_size, image_size), antialias=True),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, torch.Tensor, str]:
        row = self.manifest.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        tensor = self.transform(image)
        mask = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)
        mask_path = row["mask_path"]
        if isinstance(mask_path, str) and mask_path:
            mask_image = Image.open(mask_path).convert("L").resize(
                (self.image_size, self.image_size), resample=Image.Resampling.NEAREST
            )
            mask = torch.from_numpy((np.asarray(mask_image) > 0).astype(np.float32))[None, ...]
        return tensor, int(row["label"]), mask, str(row["domain"])
