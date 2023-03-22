from __future__ import annotations

from typing import Optional, Type

import torch
import torch.nn as nn

from ._blocks import (
    SetAttentionBlock as SAB,
    InducedSetAttentionBlock as ISAB,
    PooledMultiheadAttention as PMA,
    AttentionSchema,
)

MultilayerAttentionSchema = dict[int, AttentionSchema]


# all encoders should just change the feature dim of ptns (and their reps technically)
# but batch should stay the same


class DeepSet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        n_outputs: int,
        out_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super(DeepSet, self).__init__()
        self.n_outputs = n_outputs
        self.dim_output = out_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_outputs * self.dim_output),
        )

    def forward(
        self,
        X: torch.Tensor,
        row_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        X = self.encoder(X)

        # X: [b, p, d]
        # X.mean(-2) = mean over p dim -> [b, d]
        # ie average over all ptns for each genome
        # this part is crucial for making this set size invariant
        # but each genome also has diff number of ptns -> so need to calc number of actual ptns
        if row_mask is None:
            X = X.mean(-2)
        else:
            n_ptns = row_mask.sum(-1)
            X = X.sum(-2) / n_ptns.unsqueeze(1)
        X = self.decoder(X).reshape(-1, self.n_outputs, self.dim_output)
        return X


class SetTransformer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        n_outputs: int,
        out_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        enc_layer: Type[ISAB] | Type[SAB] = ISAB,
        dec_layer: Type[ISAB] | Type[SAB] = SAB,
        n_enc_layers: int = 2,
        n_dec_layers: int = 2,
        dropout: float = 0.0,
        bias: bool = True,
        norm: bool = True,
        **kwargs,
    ) -> None:
        super(SetTransformer, self).__init__()
        self._encoder = nn.ModuleList()
        start_dim = in_dim
        for _ in range(n_enc_layers):
            layer = enc_layer(
                in_dim=start_dim,
                out_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                bias=bias,
                norm=norm,
                **kwargs,
            )
            self._encoder.append(layer)
            start_dim = hidden_dim

        self.final_encoder_layer_idx = len(self._encoder) - 1

        self._decoder = nn.ModuleList()
        self._decoder.append(
            PMA(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                num_seeds=n_outputs,
                dropout=dropout,
                bias=bias,
                norm=norm,
            )
        )

        if issubclass(dec_layer, SAB):
            # ISAB needs num_indices
            # but SAB needs no other args
            kwargs = dict()

        for _ in range(n_dec_layers):
            layer = dec_layer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                bias=bias,
                norm=norm,
                **kwargs,
            )
            self._decoder.append(layer)

        self._decoder.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
        self._decoder.append(nn.Linear(hidden_dim, out_dim, bias=bias))

    def encode(
        self,
        X: torch.Tensor,
        return_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        return_layers: int | list[int] = -1,
    ) -> MultilayerAttentionSchema:
        if return_layers == -1:
            # default is to just return the final layer's results
            return_layers = self.final_encoder_layer_idx

        if isinstance(return_layers, int):
            return_layers = [return_layers]
        else:
            return_layers = list(return_layers)

        if self.final_encoder_layer_idx not in return_layers:
            return_layers.append(self.final_encoder_layer_idx)

        input = X
        outputs: MultilayerAttentionSchema = dict()
        for idx, layer in enumerate(self._encoder):
            output: AttentionSchema = layer(
                input, return_weights=return_weights, attn_mask=attn_mask
            )
            if idx in return_layers:
                outputs[idx] = output

            input = output.repr
        return outputs

    def decode(self, X: torch.Tensor) -> torch.Tensor:
        for layer in self._decoder:
            if isinstance(layer, (ISAB, SAB, PMA)):
                output: AttentionSchema = layer(X=X, return_weights=False)
                X = output.repr
            else:
                # final linear layers
                X = layer(X)
        return X

    def forward(
        self,
        X: torch.Tensor,
        return_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # on a regular forward pass, only the final layer is needed
        # but for debugging and model interpretation, intermediate layers
        # might be useful, esp for downstream tasks that only want the
        # encoder
        encoded_output = self.encode(
            X, return_weights=return_weights, attn_mask=attn_mask, return_layers=-1
        )
        # encoded_output should have the row-padded 0 rows as 0s still?
        Z = encoded_output[self.final_encoder_layer_idx].repr
        return self.decode(Z).squeeze()
