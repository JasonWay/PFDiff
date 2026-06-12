"""StateHaze1K / SateHaze1k paired dataset utilities.

The file is intentionally usable without PyTorch so pair validation and
manifest generation can run in lightweight environments. If torch is
available, ``StateHazeDataset`` behaves like a regular torch Dataset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import random
from typing import Iterable, List, Optional, Sequence, Tuple

from PIL import Image

try:  # Optional dependency for training environments.
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - exercised in lightweight local envs.
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
ROOT_CANDIDATES = (
    "StateHaze1K",
    "StateHaze1k",
    "SateHaze1K",
    "SateHaze1k",
    "haze1k",
)


@dataclass(frozen=True)
class PairRecord:
    """One hazy/clear image pair."""

    split: str
    key: str
    hazy_path: str
    clear_path: str

    def to_dict(self) -> dict:
        return asdict(self)


def discover_statehaze_root(search_root: str | Path = ".") -> Path:
    """Find a StateHaze1K-like directory below ``search_root``."""

    root = Path(search_root)
    candidates: List[Path] = []
    for name in ROOT_CANDIDATES:
        candidates.extend(root.glob(f"**/{name}"))
    candidates.extend(root.glob("**/haze1k_*"))

    for candidate in sorted(set(candidates), key=lambda p: (len(p.parts), str(p))):
        if _looks_like_statehaze_root(candidate):
            return candidate

    raise FileNotFoundError(
        f"Could not locate StateHaze1K/SateHaze1k under {root.resolve()}"
    )


def list_statehaze_pairs(data_root: str | Path, split: str) -> List[PairRecord]:
    """List basename-matched hazy/clear pairs for ``train``, ``val``, or ``test``."""

    data_root = Path(data_root)
    split_root = data_root / split
    hazy_dir = split_root / "hazy"
    clear_dir = split_root / "clear"
    if not hazy_dir.is_dir() or not clear_dir.is_dir():
        raise FileNotFoundError(
            f"Expected paired folders {hazy_dir} and {clear_dir} for split '{split}'"
        )

    hazy_files = _image_map(hazy_dir)
    clear_files = _image_map(clear_dir)
    missing_clear = sorted(set(hazy_files) - set(clear_files))
    missing_hazy = sorted(set(clear_files) - set(hazy_files))
    if missing_clear or missing_hazy:
        preview = {
            "missing_clear": missing_clear[:10],
            "missing_hazy": missing_hazy[:10],
        }
        raise ValueError(f"Unmatched StateHaze1K pairs in {split}: {preview}")

    return [
        PairRecord(
            split=split,
            key=key,
            hazy_path=str(hazy_files[key]),
            clear_path=str(clear_files[key]),
        )
        for key in sorted(hazy_files, key=_natural_key)
    ]


def make_train_val_split(
    train_pairs: Sequence[PairRecord],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[PairRecord], List[PairRecord]]:
    """Create a deterministic train/val split from training pairs."""

    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")
    pairs = list(train_pairs)
    indices = list(range(len(pairs)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_count = int(round(len(indices) * val_ratio))
    val_indices = set(indices[:val_count])
    train_split = [pair for idx, pair in enumerate(pairs) if idx not in val_indices]
    val_split = [pair for idx, pair in enumerate(pairs) if idx in val_indices]
    return train_split, val_split


def write_manifest(path: str | Path, pairs: Iterable[PairRecord]) -> None:
    """Write pair records to JSON for reproducible split bookkeeping."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump([pair.to_dict() for pair in pairs], handle, indent=2)


def load_rgb(path: str | Path) -> Image.Image:
    """Load an image as RGB."""

    with Image.open(path) as image:
        return image.convert("RGB")


class StateHazeDataset(Dataset):
    """PyTorch-compatible paired dataset.

    The class imports torch lazily. In a non-torch environment, construction
    still works for metadata inspection, but ``__getitem__`` raises a helpful
    error if tensor conversion is requested.
    """

    def __init__(
        self,
        pairs: Sequence[PairRecord],
        image_size: Optional[int] = None,
        training: bool = False,
    ) -> None:
        self.pairs = list(pairs)
        self.image_size = image_size
        self.training = training

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        pair = self.pairs[index]
        hazy = load_rgb(pair.hazy_path)
        clear = load_rgb(pair.clear_path)
        hazy, clear = self._transform_pair(hazy, clear)
        return {
            "key": pair.key,
            "hazy": _pil_to_tensor(hazy),
            "clear": _pil_to_tensor(clear),
            "hazy_path": pair.hazy_path,
            "clear_path": pair.clear_path,
        }

    def _transform_pair(self, hazy: Image.Image, clear: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if self.image_size:
            if self.training:
                hazy, clear = _paired_random_crop(hazy, clear, self.image_size)
                if random.random() < 0.5:
                    hazy = hazy.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    clear = clear.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            else:
                hazy = _center_crop_or_resize(hazy, self.image_size)
                clear = _center_crop_or_resize(clear, self.image_size)
        return hazy, clear


def _looks_like_statehaze_root(path: Path) -> bool:
    return (path / "train" / "hazy").is_dir() and (path / "train" / "clear").is_dir()


def _image_map(path: Path) -> dict:
    files = {}
    for item in path.iterdir():
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS:
            files[item.stem] = item
    return files


def _natural_key(value: str) -> Tuple:
    """Type-safe natural sort key for mixed alnum filenames.

    Python 3 cannot compare int and str directly while sorting tuples. Encode
    each chunk as (type_tag, value) so comparisons are always well-defined.
    """

    parts = []
    for chunk in value.replace("-", "_").split("_"):
        if chunk.isdigit():
            parts.append((0, int(chunk)))
        else:
            parts.append((1, chunk))
    return tuple(parts)


def _pil_to_tensor(image: Image.Image):
    if torch is None:
        raise RuntimeError("PyTorch is required for tensor dataset access.")
    width, height = image.size
    data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    data = data.view(height, width, 3).permute(2, 0, 1).float().div(255.0)
    return data


def _paired_random_crop(
    hazy: Image.Image,
    clear: Image.Image,
    size: int,
) -> Tuple[Image.Image, Image.Image]:
    if hazy.size != clear.size:
        raise ValueError(f"Pair size mismatch: hazy={hazy.size}, clear={clear.size}")
    width, height = hazy.size
    if width < size or height < size:
        hazy = _resize_short_side(hazy, size)
        clear = _resize_short_side(clear, size)
        width, height = hazy.size
    left = random.randint(0, width - size)
    top = random.randint(0, height - size)
    box = (left, top, left + size, top + size)
    return hazy.crop(box), clear.crop(box)


def _center_crop_or_resize(image: Image.Image, size: int) -> Image.Image:
    image = _resize_short_side(image, size)
    width, height = image.size
    left = max((width - size) // 2, 0)
    top = max((height - size) // 2, 0)
    return image.crop((left, top, left + size, top + size))


def _resize_short_side(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    short = min(width, height)
    if short == size:
        return image
    scale = size / short
    new_size = (max(size, round(width * scale)), max(size, round(height * scale)))
    return image.resize(new_size, Image.Resampling.BICUBIC)
