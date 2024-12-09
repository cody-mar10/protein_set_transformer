import torch

from pst.data.modules import GenomeDataModuleMixin
from pst.data.utils import compute_group_frequency_weights


def _add_group_weights(datamodule: GenomeDataModuleMixin):
    # scaffold weights for backwards compatibility with PST
    dataset = datamodule.dataset

    if "weight" not in dataset._registered_features:
        for name in dataset._registered_features:
            if "group" in name or "class" in name:
                group_label = dataset.get_registered_feature(name)
                break
        else:
            group_label = torch.arange(dataset.num_scaffolds)

        weights = compute_group_frequency_weights(
            group_label, datamodule.config.log_inverse
        )

        dataset.register_feature("weight", weights, feature_level="scaffold")
