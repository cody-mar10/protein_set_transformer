from platform import python_version

import lightning
import torch
from packaging import version

_VERSION_2 = version.parse("2")

TORCH_VERSION_GTE_2 = version.parse(torch.__version__) >= _VERSION_2
LIGHTNING_VERSION_GTE_2 = version.parse(lightning.__version__) >= _VERSION_2

PYTHON_VERSION_GTE_3_11 = version.parse(python_version()) >= version.parse("3.11")
