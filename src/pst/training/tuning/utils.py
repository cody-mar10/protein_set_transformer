from __future__ import annotations

from pathlib import Path

import numpy as np
import tables as tb


def _peek_feature_dim(data_file: Path) -> int:
    with tb.File(data_file) as fp:
        return np.shape(fp.root.data[0])[-1]
