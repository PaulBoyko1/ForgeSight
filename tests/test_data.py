from pathlib import Path

import numpy as np
from PIL import Image

from forgesight.data.mvtec import ManifestDataset, discover_category


def _image(path: Path, value: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((16, 16, 3), value, dtype=np.uint8)).save(path)


def test_discover_category_and_dataset(tmp_path: Path) -> None:
    root = tmp_path / "can"
    _image(root / "train" / "good" / "000.png")
    _image(root / "validation" / "good" / "001.png")
    _image(root / "test_public" / "good" / "002.png")
    _image(root / "test_public" / "dent" / "003.png")
    _image(root / "ground_truth" / "dent" / "003_mask.png", value=255)

    manifest = discover_category(root)
    assert len(manifest) == 4
    anomaly = manifest[manifest["label"] == 1].iloc[0]
    assert anomaly["defect_type"] == "dent"
    assert anomaly["mask_path"].endswith("003_mask.png")

    dataset = ManifestDataset(manifest[manifest["label"] == 1], image_size=32)
    image, label, mask, domain = dataset[0]
    assert image.shape == (3, 32, 32)
    assert label == 1
    assert mask.sum() == 32 * 32
    assert domain == "default"
