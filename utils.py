"""Small, dependency-light helpers shared across the pipeline."""

from __future__ import annotations

import hashlib
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np


# ---------------------------------------------------------------------------
# Filesystem / serialisation
# ---------------------------------------------------------------------------


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Any, path: Path, indent: int = 2) -> None:
    ensure_dir(Path(path).parent)
    def _default(o: Any):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(f"not JSON serialisable: {type(o)!r}")

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, default=_default)
        fh.write("\n")


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def fmt_bytes(n: int | float) -> str:
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TiB"


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def imread_unicode(path: Path, flags: int | None = None) -> np.ndarray | None:
    """``cv2.imread`` that survives non-ASCII paths on Windows.

    ``cv2.imread`` passes the filename through the ANSI code page and fails
    silently on Unicode paths; decoding bytes read by Python avoids that.
    Returns ``None`` when the file cannot be decoded, matching ``cv2.imread``.
    """
    import cv2

    if flags is None:
        flags = cv2.IMREAD_COLOR
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flags)


def dhash(gray: np.ndarray, hash_size: int = 8) -> int:
    """Difference hash of a grayscale image.

    Returns a ``hash_size**2``-bit integer; for the default 8 it is a 64-bit
    descriptor whose Hamming distance approximates perceptual distance.
    The image is resized to ``(hash_size + 1, hash_size)`` and neighbouring
    horizontal pixels are compared.
    """
    import cv2

    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = diff.flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return value


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_matrix(values: np.ndarray, block: int = 512) -> np.ndarray:
    """Full pairwise Hamming distance matrix for uint64 hash values.

    Vectorised: each row is XORed against every column and the 8 byte-planes are
    popcounted with a lookup table.  A naive Python double loop over 6,150
    images (18.9 M pairs) takes ~40 s; this takes well under a second.
    """
    v = np.asarray(values, dtype=np.uint64)
    n = v.size
    out = np.zeros((n, n), dtype=np.uint8)
    bytes_view = v.view(np.uint8).reshape(n, 8)  # little-endian byte planes
    for start in range(0, n, block):
        stop = min(start + block, n)
        x = np.bitwise_xor(v[start:stop, None], v[None, :])
        xb = x.view(np.uint8).reshape(stop - start, n, 8)
        out[start:stop] = _POPCOUNT[xb].sum(axis=2, dtype=np.uint16).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


@contextmanager
def timer(label: str = "task") -> Iterator[dict[str, float]]:
    """``with timer("audit") as t: ...`` then ``t["elapsed"]``."""
    box: dict[str, float] = {}
    start = time.perf_counter()
    yield box
    box["elapsed"] = time.perf_counter() - start
    print(f"[{label}] {box['elapsed']:.2f}s")


def human_pct(part: float, whole: float) -> str:
    return f"{100.0 * part / whole:.2f}%" if whole else "n/a"
