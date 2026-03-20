from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def expected_stage1_tensor_shape(image_size: Sequence[int]) -> tuple[int, int, int]:
    height, width = (int(value) for value in image_size)
    return (height, width, 3)


def load_stage1_npy_tensor(
    path_value: str | Path | bytes | np.bytes_,
    image_size: Sequence[int],
) -> np.ndarray:
    if isinstance(path_value, (bytes, bytearray, np.bytes_)):
        resolved_path = Path(path_value.decode("utf-8"))
    else:
        resolved_path = Path(str(path_value))

    tensor = np.load(resolved_path, allow_pickle=False)
    tensor = np.asarray(tensor, dtype=np.float32)
    expected_shape = expected_stage1_tensor_shape(image_size)
    if tensor.shape != expected_shape:
        raise ValueError(
            f"Expected Stage 1 tensor at {resolved_path} to have shape "
            f"{expected_shape}, got {tensor.shape}."
        )
    return np.clip(tensor, 0.0, 1.0).astype(np.float32, copy=False)
