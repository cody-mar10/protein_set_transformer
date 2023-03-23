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
        self.out_dim = out_dim
        self._encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self._decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_outputs * self.out_dim),
        )

    def encode(
        self, X: torch.Tensor, row_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # X: [b, p, d]
        # X.mean(-2) = mean over p dim -> [b, d]
        # ie average over all ptns for each genome
        # this part is crucial for making this set size invariant
        # but each genome also has diff number of ptns -> so need to calc number of actual ptns
        X = self._encoder(X)
        if row_mask is None:
            X = X.mean(-2)
        else:
            n_items = row_mask.sum(-1)
            X = X.sum(-2) / n_items.unsqueeze(1)

        return X

    def decode(self, X: torch.Tensor) -> torch.Tensor:
        return self._decoder(X).reshape(-1, self.n_outputs, self.out_dim)

    def forward(
        self,
        X: torch.Tensor,
        row_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.decode(self.encode(X, row_mask=row_mask))


LayerType = Type[ISAB] | Type[SAB]


class SetTransformer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        n_outputs: int,
        out_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_indices: int = 32,
        n_enc_layers: int = 2,
        n_dec_layers: int = 2,
        dropout: float = 0.0,
        bias: bool = True,
        norm: bool = True,
    ) -> None:
        """A SetTransformer implementation that uses an encoder-decoder framework.

        The encoder represents set elements to be most similar within a set and
        identifies set elements that are similar between sets.

        The decoder decodes individual set element representations to encode
        a set-level representation that also respects set-set relationships.

        Args:
            in_dim (int): data feature dimension
            n_outputs (int): number of outputs
            out_dim (int): data output embedding dimension
            hidden_dim (int, optional): dimension of the hidden layers. Defaults to 128.
            num_heads (int, optional): number of attention heads. Defaults to 4.
            num_indices (int, optional): projection dimension for large set efficient multiheaded attention. Defaults to 32.
            n_enc_layers (int, optional): number of encoder layers. Defaults to 2.
            n_dec_layers (int, optional): number of decoder layers, not including a pooling attention layer at the beginning and the fully connected layers at the end. Defaults to 2.
            dropout (float, optional): dropout probability during training. Defaults to 0.0.
            bias (bool, optional): Include bias in linear layers. Defaults to True.
            norm (bool, optional): Include a LayerNorm operation after each attention block. Defaults to True.
        """
        super(SetTransformer, self).__init__()
        self._encoder = nn.ModuleList()
        start_dim = in_dim
        for _ in range(n_enc_layers):
            layer = ISAB(
                in_dim=start_dim,
                out_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                bias=bias,
                norm=norm,
                num_indices=num_indices,
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

        for _ in range(n_dec_layers):
            layer = SAB(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                bias=bias,
                norm=norm,
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
