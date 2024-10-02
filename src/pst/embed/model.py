from __future__ import annotations

from typing import Literal, cast

import esm
import lightning as L
import torch

BatchType = tuple[list[str], list[str], torch.Tensor]


# Prediction only
class ESM2(L.LightningModule):
    MODELS = {
        "esm2_t48_15B": esm.pretrained.esm2_t48_15B_UR50D,
        "esm2_t36_3B": esm.pretrained.esm2_t36_3B_UR50D,
        "esm2_t33_650M": esm.pretrained.esm2_t33_650M_UR50D,
        "esm2_t30_150M": esm.pretrained.esm2_t30_150M_UR50D,
        "esm2_t12_35M": esm.pretrained.esm2_t12_35M_UR50D,
        "esm2_t6_8M": esm.pretrained.esm2_t6_8M_UR50D,
    }

    MODELVALUES = Literal[
        "esm2_t48_15B",
        "esm2_t36_3B",
        "esm2_t33_650M",
        "esm2_t30_150M",
        "esm2_t12_35M",
        "esm2_t6_8M",
    ]

    LAYERS_TO_MODELNAME = {
        6: "esm2_t6_8M",
        48: "esm2_t48_15B",
        36: "esm2_t36_3B",
        33: "esm2_t33_650M",
        30: "esm2_t30_150M",
        12: "esm2_t12_35M",
    }

    def __init__(self, model: esm.ESM2, alphabet: esm.Alphabet) -> None:
        super().__init__()
        self.model = model
        self.alphabet = alphabet
        self.repr_layers = model.num_layers

    def predict_step(
        self, batch: BatchType, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        labels, seqs, tokens = batch
        seqlens = torch.sum(tokens != self.alphabet.padding_idx, dim=1)
        results = self.model(
            tokens, repr_layers=[self.repr_layers], return_contacts=False
        )
        token_repr: torch.Tensor = results["representations"][self.repr_layers]

        # Generate sequence level representations by averaging over token repr
        # NOTE: token 0 is beginning-of-seq token
        seq_rep = torch.vstack(
            [
                token_repr[i, 1 : token_lens - 1].mean(dim=0)
                for i, token_lens in enumerate(seqlens)
            ]
        )
        return seq_rep

    @classmethod
    def from_model_name(cls, model_name: ESM2.MODELVALUES) -> "ESM2":
        model_loader = ESM2.MODELS[model_name]
        esm_model, alphabet = model_loader()
        esm_model = cast(esm.ESM2, esm_model)
        return cls(model=esm_model, alphabet=alphabet)
