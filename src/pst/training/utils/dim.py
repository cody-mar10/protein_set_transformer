import numpy as np
import tables as tb

from pst.typing import FilePath
from pst.utils.cli.modes import TrainingMode


def peak_feature_dim(data_file: FilePath) -> int:
    with tb.File(data_file) as fp:
        return np.shape(fp.root.data[0])[-1]


def check_feature_dim(config: TrainingMode):
    if config.model.in_dim == -1:
        config.model.in_dim = peak_feature_dim(config.data.file)

    if config.model.out_dim == -1:
        config.model.out_dim = config.model.in_dim
