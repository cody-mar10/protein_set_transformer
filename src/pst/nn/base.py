import sys
from collections.abc import Mapping
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Generic, Literal, Optional, TypeVar, Union, cast

if sys.version_info >= (3, 11):  # pragma: >=3.11 cover
    from typing import Self
else:  # pragma: <3.11 cover
    from typing_extensions import Self

import lightning as L
import torch
from transformers import get_linear_schedule_with_warmup

from pst.data.modules import GenomeDataset
from pst.nn.config import BaseModelConfig
from pst.nn.layers import PositionalEmbedding
from pst.nn.models import SetTransformer, SetTransformerDecoder, SetTransformerEncoder
from pst.typing import (
    EdgeAttnOutput,
    GenomeGraphBatch,
    GraphAttnOutput,
    MaskedGenomeGraphBatch,
    OptTensor,
)
from pst.utils._signatures import _resolve_config_type_from_init

_STAGE_TYPE = Literal["train", "val", "test"]
_FIXED_POINTSWAP_RATE = "FIXED_POINTSWAP_RATE"
_ModelT = TypeVar("_ModelT", SetTransformer, SetTransformerEncoder)
_BaseConfigT = TypeVar("_BaseConfigT", bound=BaseModelConfig)


class PositionalStrandEmbeddingModule(L.LightningModule):
    def __init__(self, in_dim: int, embed_scale: int, max_size: int):
        super().__init__()

        self.max_size = max_size
        embedding_dim = in_dim // embed_scale

        self.positional_embedding = PositionalEmbedding(dim=embedding_dim, max_size=max_size)

        # embed +/- gene strand
        self.strand_embedding = torch.nn.Embedding(num_embeddings=2, embedding_dim=embedding_dim)

        self.extra_embedding_dim = 2 * embedding_dim

    def _convert_strand_if_needed(self, strand: torch.Tensor) -> torch.Tensor:
        """Convert the strand tensor from bool to long if needed. This is necessary for indexing
        the strand embeddings."""

        if strand.dtype != torch.long:
            return strand.long()
        return strand

    def internal_embeddings(
        self, batch: GenomeGraphBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the strand and positional embeddings for the proteins in the batch.

        Then return the embeddings for the strand, the positional embeddings, and the concatenated
        embeddings of the proteins with the positional and strand embeddings.

        Args:
            batch (GenomeGraphBatch): The batch object containing the protein embeddings,
                edge index, the index pointer, the number of proteins in each genome, etc, that
                are used for the forward pass of the `SetTransformer` or `SetTransformerEncoder`.
                This object models the data patterns of PyTorch Geometric graphs.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                (concatenated embeddings, positional embeddings, strand embeddings)
        """
        strand = self._convert_strand_if_needed(batch.strand)
        strand_embed = self.strand_embedding(strand)
        positional_embed = self.positional_embedding(batch.pos.squeeze())

        x_cat = self.concatenate_embeddings(
            x=batch.x, positional_embed=positional_embed, strand_embed=strand_embed
        )

        return x_cat, positional_embed, strand_embed

    def masked_embeddings(
        self, batch: MaskedGenomeGraphBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_mask = batch.node_mask
        strand = batch.strand[node_mask]
        strand = self._convert_strand_if_needed(strand)
        pos = batch.pos.squeeze()[node_mask]
        x = batch.masked_embeddings

        # ok to keep attached to gradient graph since we
        # care more about protein embedding identity rather than these embeddings
        # since we already would've known the position at least
        strand_embed: torch.Tensor = self.strand_embedding(strand)
        positional_embed: torch.Tensor = self.positional_embedding(pos)

        x_cat = self.concatenate_embeddings(
            x=x,
            positional_embed=positional_embed,
            strand_embed=strand_embed,
        )

        return x_cat, positional_embed, strand_embed

    def concatenate_embeddings(
        self,
        x: torch.Tensor,
        positional_embed: torch.Tensor,
        strand_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate the protein embeddings with the positional and strand embeddings. The
        output tensor will have shape [num_proteins, in_dim + 2 * embedding_dim]. The order
        of the concatenation is [protein embeddings, positional embeddings, strand embeddings].

        Args:
            x (torch.Tensor): protein embeddings, shape: [num_proteins, in_dim]
            positional_embed (torch.Tensor): positional embeddings for each protein, shape:
                [num_proteins, embedding_dim]
            strand_embed (torch.Tensor): strand embeddings for each protein, shape:
                [num_proteins, embedding_dim]

        Returns:
            torch.Tensor: concatenated embeddings, shape: [num_proteins, in_dim + 2 * embedding_dim],
                order: [protein embeddings, positional embeddings, strand embeddings]
        """
        x_cat = torch.cat((x, positional_embed, strand_embed), dim=-1)
        return x_cat


class _BaseProteinSetTransformer(PositionalStrandEmbeddingModule, Generic[_ModelT, _BaseConfigT]):
    """Base class for `ProteinSetTransformer` models for either genome-level or protein-level
    tasks. This class sets up either the underlying `SetTransformer` model (genome) or the
    `SetTransformerEncoder` (protein) along with the positional and strand embeddings.

    Further, the `lightning` specific methods for training, validation, and testing are
    implemented by simply calling the `forward` method of this class.

    This is meant to be an abstract base class, so subclasses must implement the following
    methods:

    1. `setup_objective`: to setup the loss function. For loss functions that maintain state,
            like the margin in a triplet loss, a custom model config subclass of `BaseModelConfig`
            that updates the `BaseLossConfig` can be used to add new arguments to this function.
            All fields in the `BaseModelConfig.loss` field are passed to this function.
    2. `forward`: to define the forward pass of the model, including how the loss is computed
    3. `forward_step`: to define how data is passed to the underlying models. This method is
            called by the `databatch_forward` method, which unwraps a custom batch object to pass
            to the model. Presumably, the `forward` method calls the `databatch_forward` or the
            `forward_step` method directly.

    Further, since this is a `lightning.LightningModule`, you can override any of the
    lightning methods such as `training_step`, `validation_step`, `test_step`,
    `configure_optimizers` to further customize functionality.

    WARNING: Subclasses should NOT change the name of the config argument in the __init__ method.
    They should always use `config` as the first argument that is TYPE-HINTED as a subclass of
    `BaseModelConfig`. This is necessary for the `from_pretrained` classmethod to work correctly
    for loading pretrained models.

    This is not meant to be directly subclassed by users. Instead, users should subclass
    `BaseProteinSetTransformer` or `BaseProteinSetTransformerEncoder` for genome-level or
    protein-level tasks. Note: the `BaseProteinSetTransformer` can also be used for dual
    genome and protein-level tasks.
    """

    # keep track of all the valid pretrained model names
    PRETRAINED_MODEL_NAMES = set()
    config: _BaseConfigT

    # NOTE: do not change the name of the config var in all subclasses
    def __init__(self, config: _BaseConfigT, model_type: type[_ModelT]) -> None:
        expected_config_type = self._resolve_model_config_type()
        if not isinstance(config, expected_config_type):
            raise ValueError(
                f"Model config {config} is not the expected type {expected_config_type}"
            )

        super().__init__(
            in_dim=config.in_dim,
            embed_scale=config.embed_scale,
            max_size=config.max_proteins,
        )

        if config.out_dim == -1:
            config.out_dim = config.in_dim

        # only needed for saving/loading models
        self.original_config = config

        # deep copy so that embedding size changes do not affect the original config
        # this really only matters for interactive sessions
        self.config = config.clone(deep=True)

        # need all new models to set _FIXED_POINTSWAP_RATE to True
        # this way we know that saved models are using the correct sample rate
        # NOTE: regarding changes to embed dim, this should save the initial values
        # BEFORE the dimension changes
        self.save_hyperparameters(
            {_FIXED_POINTSWAP_RATE: True, "config": self.original_config.to_dict()}
        )

        self.config.in_dim += self.extra_embedding_dim

        if not config.proj_cat:
            # plm embeddings, positional embeddings, and strand embeddings
            # will be concatenated together and then projected back to the original dim
            # by the first attention layer, BUT if we don't want that
            # then the output dimension will be equal to the original feature dim
            # plus the dim for both the positional and strand embeddings
            self.config.out_dim = self.config.in_dim

        # for genomic PST, the model is a SetTransformer
        # for protein PST, the model is a SetTransformerEncoder
        self.model = self.setup_model(model_type)
        self.is_genomic = isinstance(self.model, SetTransformer)

        self.criterion = self.setup_objective(**self.config.loss.to_dict())

    def _setup_model(
        self, model_type: type[_ModelT], include: Optional[set[str]] = None, **kwargs
    ) -> _ModelT:
        model = model_type(**self.config.to_dict(include=include), **kwargs)
        if self.config.compile:
            model: _ModelT = torch.compile(model)  # type: ignore

        return model

    def setup_model(self, model_type: type[_ModelT]) -> _ModelT:
        # for some reason, typehinting hates if this is part of the if directly
        condition = issubclass(model_type, SetTransformer)

        if condition:
            # genome PST
            include = {
                "in_dim",
                "out_dim",
                "num_heads",
                "n_enc_layers",
                "dropout",
                "layer_dropout",
            }
            kwargs = {}
        else:
            # protein PST
            include = {"in_dim", "out_dim", "num_heads", "dropout", "layer_dropout"}
            kwargs = {"n_layers": self.config.n_enc_layers}

        return self._setup_model(model_type, include=include, **kwargs)

    def setup_objective(self, **kwargs) -> torch.nn.Module | Callable[..., torch.Tensor]:
        """**Must be overridden by subclasses to setup the loss function.**

        This function is always passed all fields on the `LossConfig` defined as a subfield of
        the `ModelConfig` used to instantiate this class. For loss functions that maintain
        state, like the margin in a triplet loss, the `LossConfig` (and subsequently the
        `ModelConfig`) can be overridden to add new arguments to this function.

        This funciton should return a callable, such as a `torch.nn.Module`, whose `__call__`
        method computes the loss.
        """
        raise NotImplementedError

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.config.optimizer.lr,
            betas=self.config.optimizer.betas,
            weight_decay=self.config.optimizer.weight_decay,
        )
        config: dict[str, Any] = {"optimizer": optimizer}
        if self.config.optimizer.use_scheduler:
            if self.fabric is None:
                self.estimated_steps = self.trainer.estimated_stepping_batches

            # TODO: i dont think self.estimated_steps exists without a trainer?
            scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self.config.optimizer.warmup_steps,
                num_training_steps=self.estimated_steps,
            )
            config["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }

        return config

    def check_max_size(self, dataset: GenomeDataset):
        """Checks the maximum number of proteins encoded in a single genome from the dataset.
        The positional embeddings have a maximum size that cannot be easily expanded due to the
        need to readjust the internal graph representations of each genome.

        Args:
            dataset (GenomeDataset): the genome graph dataset
        """
        if dataset.max_size > self.positional_embedding.max_size:
            self.positional_embedding.expand(dataset.max_size)

    @cached_property
    def encoder(self) -> SetTransformerEncoder:
        if isinstance(self.model, SetTransformer):
            return self.model.encoder
        return self.model

    @cached_property
    def decoder(self) -> SetTransformerDecoder:
        if isinstance(self.model, SetTransformer):
            return self.model.decoder
        raise AttributeError("SetTransformerEncoder does not have a decoder")

    def log_loss(self, loss: torch.Tensor, batch_size: int, stage: _STAGE_TYPE):
        """Simple wrapper around the lightning.LightningModule.log method to log the loss."""
        self.log(
            f"{stage}_loss",
            value=loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )

    def _loss_step(
        self,
        batch: GenomeGraphBatch,
        stage: _STAGE_TYPE,
        **kwargs,
    ) -> torch.Tensor:
        loss = self(batch=batch, **kwargs)
        # TODO: is this correct for batch size?
        batch_size = batch.num_proteins.numel()
        # if fabric attached, it will handle logging
        if self.fabric is None:
            self.log_loss(loss, batch_size, stage)

        return loss

    ### lightning.Trainer methods ###

    def training_step(
        self, train_batch: GenomeGraphBatch, batch_idx: int, **kwargs
    ) -> torch.Tensor:
        return self._loss_step(train_batch, stage="train", **kwargs)

    def validation_step(
        self, val_batch: GenomeGraphBatch, batch_idx: int, **kwargs
    ) -> torch.Tensor:
        return self._loss_step(batch=val_batch, stage="val", **kwargs)

    def test_step(self, test_batch: GenomeGraphBatch, batch_idx: int, **kwargs) -> torch.Tensor:
        return self(batch=test_batch, stage="test", **kwargs)

    def _databatch_forward_with_embeddings(
        self, batch: GenomeGraphBatch, return_attention_weights: bool = True
    ) -> GraphAttnOutput | EdgeAttnOutput:
        x_with_pos_and_strand, _, _ = self.internal_embeddings(batch)

        return self.databatch_forward(
            batch=batch,
            x=x_with_pos_and_strand,
            return_attention_weights=return_attention_weights,
        )

    def predict_step(
        self, batch: GenomeGraphBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> GraphAttnOutput | EdgeAttnOutput:
        return self._databatch_forward_with_embeddings(batch=batch, return_attention_weights=True)

    ### End lightning.Trainer methods ###

    def forward_step(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        ptr: torch.Tensor,
        batch: OptTensor = None,
        node_mask: OptTensor = None,
        return_attention_weights: bool = False,
    ) -> GraphAttnOutput | EdgeAttnOutput:
        """**Must be overridden by subclasses to define how the model's direct forward pass
        uses this these arguments.**

        This is a simple wrapper around the model's forward method used simply for computing
        the embeddings.

        This method should be called by the `.forward` method. Additionally, the specific data
        handling steps to compute the loss should be handled by the `.forward` method.

        These arguments are used by the graph data model of PyTorch Geometric.

        Args:
            x (torch.Tensor): shape [num_proteins, in_dim]; the protein embeddings
            edge_index (torch.Tensor): shape (2, num_edges); the edge index tensor that connects
                protein nodes in genome subgraphs
            ptr (torch.Tensor): shape (num_proteins + 1,); the index pointer tensor for efficient
                random access to the protein embeddings from each genome. Only needed for genome
                level implementations, such as with the decoder of the `SetTransformer`.
            batch (Optional[torch.Tensor]): shape (num_proteins,); the batch tensor that assigns
                each protein to a specific genome (graph). This can be computed from the `ptr`
                tensor, so it is optional, but passing a precomputed batch tensor will be more
                efficient.
            node_mask (Optional[torch.Tensor]): Used to mask out certain nodes in the graph,
                such as during masked language modeling, during node-node attention calculation.
                If not provided, no masking is done.
            return_attention_weights (bool): whether to return the attention weights, probably
                only wanted for debugging or for final predictions

        Returns:
            NamedTuples:
                - GraphAttnOutput: IF this is a genomic PST (using the `SetTransformer` internally),
                    the output includes fields (out, attn) where `out` is the genome embeddings
                    and `attn` is the attention weights (shape: [num_genomes, num_heads]). The
                    attention weights are only returned if `return_attention_weights` is True.
                - EdgeAttnOutput: IF this is a protein PST (using the `SetTransformerEncoder`
                    internally), the output includes fields (out, edge_index, attn) where `out`
                    is the protein embeddings, `edge_index` is the edge index tensor used for
                    the message passing attention calculation, and `attn` is the attention
                    weights (shape: [num_edges, num_heads]). The attention weights are only
                    returned if `return_attention_weights` is True. NOTE: the `edge_index`
                    tensor is not necessarily the same as the one that was input since the
                    model internally adds self loops if they are not present. `attn` is computed
                    over the edges in the **returned** `edge_index`.
        """
        raise NotImplementedError  # TODO: should the PST decoder accept a node mask?

    def databatch_forward(
        self,
        batch: GenomeGraphBatch,
        node_mask: OptTensor = None,
        return_attention_weights: bool = False,
        x: OptTensor = None,
    ) -> GraphAttnOutput | EdgeAttnOutput:
        """Calls the forward method of the underlying `SetTransformer` or `SetTransformerEncoder`
        model by unwrapping the `GenomeGraphBatch` fields.

        Args:
            batch (GenomeGraphBatch): The batch object containing the protein embeddings,
                edge index, the index pointer, the number of proteins in each genome, etc, that
                are used for the forward pass of the `SetTransformer` or `SetTransformerEncoder`.
                This object models the data patterns of PyTorch Geometric graphs.
            node_mask (Optional[torch.Tensor]): Used to mask out certain nodes in the graph,
                such as during masked language modeling, during node-node attention calculation.
                If not provided, no masking is done.
            return_attention_weights (bool): whether to return the attention weights, probably
                only wanted for debugging or for final predictions
            x (Optional[torch.Tensor]): Used to allow custom protein embeddings, such as those
                that include positional and strand embeddings, to be passed to the model. If
                `x` is not provided, the model will use the raw protein embeddings in the batch
                object. This is useful when both the raw protein embeddings and modified
                embeddings are needed for the forward pass.

        Returns:
            NamedTuples:
                - GraphAttnOutput: IF this is a genomic PST (using the `SetTransformer` internally),
                    the output includes fields (out, attn) where `out` is the genome embeddings
                    and `attn` is the attention weights. The attention weights are only returned
                    if `return_attention_weights` is True.
                - EdgeAttnOutput: IF this is a protein PST (using the `SetTransformerEncoder`
                    internally), the output includes fields (out, edge_index, attn) where `out`
                    is the protein embeddings, `edge_index` is the edge index tensor used for
                    the message passing attention calculation, and `attn` is the attention
                    weights. The attention weights are only returned if `return_attention_weights`
                    is True. NOTE: the `edge_index` tensor is not necessarily the same as the
                    one that was input since the model internally adds self loops if they are
                    not present.
        """
        if x is None:
            x = batch.x

        return self.forward_step(
            x=x,
            edge_index=batch.edge_index,
            ptr=batch.ptr,
            batch=batch.batch,
            node_mask=node_mask,
            return_attention_weights=return_attention_weights,
        )

    def forward(self, batch: GenomeGraphBatch) -> torch.Tensor:
        """**Must be overridden by subclasses to define the forward pass.**

        Implement the specific steps for the forward pass to compute loss. For example, if
        using a triplet loss objective, this method should compute the genome embeddings by
        calling the `.forward_step` method, then triplet sampling, and then computing the loss.

        An optional but recommended method to call instead of `.forward_step` is
        `.databatch_forward`, which can take the batch object itself, and unwrap the arguments
        to `.forward_step`.

        Args:
            batch (GenomeGraphBatch): The batch object containing the protein embeddings,
                edge index, the index pointer, the number of proteins in each genome, etc, that
                are used for the forward pass of the `SetTransformer`. This object models the
                data patterns of PyTorch Geometric graphs.

        Returns:
            torch.Tensor: the loss tensor attached to the gradient graph
        """
        raise NotImplementedError

    @classmethod
    def _resolve_model_config_type(cls) -> type[_BaseConfigT]:
        model_config_type = _resolve_config_type_from_init(
            cls, config_name="config", default=BaseModelConfig
        )
        return cast(type[_BaseConfigT], model_config_type)

    def _try_load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True):
        try:
            # just try to load directly
            self.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            # if that fails, then we need to try to see if we are loading a pretrained model
            # that does not have values for new layers in subclassed models

            # get the base parameters of the SetTransformer or SetTransformerEncoder
            # along with the positional and strand embeddings
            base_params = {f"model.{name}" for name, _ in self.model.named_parameters()}

            # PositionalStrandEmbeddingModuleMixin params
            for embedding_name in ("positional_embedding", "strand_embedding"):
                layer: torch.nn.Module = getattr(self, embedding_name)
                for name, _ in layer.named_parameters():
                    base_params.add(f"{embedding_name}.{name}")

            # get all new params
            current_params = {name for name, _ in self.named_parameters()}

            new_params = current_params - base_params

            # now try to load the state dict
            missing, unexpected = map(set, self.load_state_dict(state_dict, strict=False))

            # missing should be equivalent to the new params if loaded correctly
            still_missing = new_params - missing

            if still_missing:
                raise RuntimeError(
                    f"Missing parameters: {still_missing} when loading the state dict"
                )

            if strict and unexpected:
                raise RuntimeError(
                    f"Unexpected parameters: {unexpected} when loading the state dict"
                )

    @staticmethod
    def _adjust_checkpoint_inplace(ckpt: dict[str, Any]):
        # no-op for base classes, subclasses can override this to adjust the checkpoint
        return

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        model_type: type[_ModelT],
        model_config_type: Optional[type[_BaseConfigT]] = None,
        strict: bool = True,
        **update_kwargs,
    ) -> Self:
        """Load a model from a pretrained (or just trained) checkpoint.

        This method handles subclasses that add new trainable parameters that want to start
        from a parent pretrained model. For example, subclassing a `ProteinSetTransformer` model
        and changing the objective from triplet loss to some label classification would introduce
        new trainable layers that would benefit from starting from a pretrained PST.

        This method works with allows interchanging between `SetTransformer` (genomic) and
        `SetTransformerEncoder` (protein) models. The most common case of this would be with
        a pretrained genomic PST that is then finetuned for a protein-level task in which
        the decoder of the `SetTransformer` is not used.

        Args:
            pretrained_model_name_or_path (str | Path): name or file path to the pretrained model
                checkpoint. NOTE: passing model names is not currently supported
            model_type (Type[_ModelT]): underlying model type. For protein-level tasks, this
                should be `SetTransformerEncoder`. For genome-level tasks, this should be
                `SetTransformer`. It is recommended that subclasses handle this so users do not
                have to pass this argument.
            model_config_type (Optional[Type[_BaseConfigT]], optional): the model config
                pydantic model. If passed, this must be a subclass of `BaseModelConfig`.
                Defaults to None, meaning that it will be auto-detected from the class's
                __init__ type hinted signature or default to `BaseModelConfig` if the
                type annotation cannot be detected. NOTE: it is recommended that subclasses
                type hint the `config` argument to the __init__ method to ensure that the
                type of the model config is correctly detected.
            strict (bool, optional): raise a RuntimeError if there are unexpected parameters
                in the checkpoint's state dict. Defaults to True.
            **update_kwargs: additional keyword arguments to update the model config with

        Raises:
            NotImplementedError: Loading models from their names is not implemented yet
        """
        if not isinstance(pretrained_model_name_or_path, Path):
            # could either be a str path or a model name
            valid_model_names = cls.PRETRAINED_MODEL_NAMES

            if pretrained_model_name_or_path in valid_model_names:
                # load from external source
                # TODO: can call download code written in this module...
                raise NotImplementedError("Loading from external source not implemented yet")
            else:
                # assume str is a file path
                pretrained_model_name_or_path = Path(pretrained_model_name_or_path)

        if model_config_type is None:
            model_config_type = cls._resolve_model_config_type()

        ckpt = torch.load(pretrained_model_name_or_path, map_location="cpu")
        cls._adjust_checkpoint_inplace(ckpt)

        # need to merge ckpt["hyper_parameters"] with update_kwargs with nested dicts
        # however, there should only be 2 nesting levels if present

        hparams = ckpt["hyper_parameters"]

        if "config" in hparams:
            hparams = hparams["config"]

        for key, value in update_kwargs.items():
            if isinstance(value, Mapping):
                for subkey, subvalue in value.items():
                    hparams[key][subkey] = subvalue
            else:
                hparams[key] = value

        model_config = model_config_type.from_dict(hparams)

        try:
            model = cls(config=model_config, model_type=model_type)
        except TypeError:
            # subclasses may not have the model_type as it will probably be set
            model = cls(config=model_config)  # type: ignore

        model._try_load_state_dict(ckpt["state_dict"], strict=strict)

        return model


class BaseProteinSetTransformer(_BaseProteinSetTransformer[SetTransformer, _BaseConfigT]):
    """Base class for a genome-level `ProteinSetTransformer` model. This class sets up the
    the underlying `SetTransformer` model along with the positional and strand embeddings.

    This is an abstract base class, so subclasses must implement the following methods:
    1. `setup_objective`: to setup the loss function
    2. `forward`: to define the forward pass of the model, including how the loss is computed

    If the loss function requires additional parameters, a custom model config subclass of `BaseModelConfig` can be used that replaces the `BaseLossConfig` with new fields.
    """

    MODEL_TYPE = SetTransformer

    def __init__(self, config: _BaseConfigT):
        super().__init__(config=config, model_type=self.MODEL_TYPE)

    # change type annotations in the signature
    def databatch_forward(
        self,
        batch: GenomeGraphBatch,
        node_mask: OptTensor = None,
        return_attention_weights: bool = False,
        x: OptTensor = None,
    ) -> GraphAttnOutput:
        result = super().databatch_forward(
            batch=batch,
            node_mask=node_mask,
            return_attention_weights=return_attention_weights,
            x=x,
        )

        return cast(GraphAttnOutput, result)

    def predict_step(
        self, batch: GenomeGraphBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> GraphAttnOutput:
        result = super().predict_step(
            batch=batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx
        )

        return cast(GraphAttnOutput, result)

    def forward_step(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        ptr: torch.Tensor,
        batch: OptTensor = None,
        node_mask: OptTensor = None,
        return_attention_weights: bool = False,
    ) -> GraphAttnOutput:
        output: GraphAttnOutput = self.model(
            x=x,
            edge_index=edge_index,
            ptr=ptr,
            batch=batch,
            node_mask=node_mask,
            return_attention_weights=return_attention_weights,
        )

        return output

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        strict: bool = True,
        **update_kwargs,
    ) -> Self:
        """Load a model from a pretrained (or just trained) checkpoint.

        This method handles subclasses that add new trainable parameters that want to start
        from a parent pretrained model. For example, subclassing a `ProteinSetTransformer` model
        and changing the objective from triplet loss to some label classification would introduce
        new trainable layers that would benefit from starting from a pretrained PST.

        This method works with allows interchanging between `SetTransformer` (genomic) and
        `SetTransformerEncoder` (protein) models. The most common case of this would be with
        a pretrained genomic PST that is then finetuned for a protein-level task in which
        the decoder of the `SetTransformer` is not used.

        Args:
            pretrained_model_name_or_path (str | Path): name or file path to the pretrained model
                checkpoint. NOTE: passing model names is not currently supported
            strict (bool, optional): raise a RuntimeError if there are unexpected parameters
                in the checkpoint's state dict. Defaults to True.
            **update_kwargs: additional keyword arguments to update the model config

        Raises:
            NotImplementedError: Loading models from their names is not implemented yet
        """
        return super().from_pretrained(
            pretrained_model_name_or_path,
            model_type=cls.MODEL_TYPE,
            strict=strict,
            **update_kwargs,
        )


######### Protein-level models #########


class BaseProteinSetTransformerEncoder(
    _BaseProteinSetTransformer[SetTransformerEncoder, _BaseConfigT],
):
    """Base class for protein-level tasks using the `SetTransformerEncoder` model. This class
    can also be derived from pretrained genome-level models, such as the `ProteinSetTransformer`.
    The genome decoding layers from pretrained models are dropped, leaving on the encoder layers
    and the positional and strand embeddings.

    This class is meant to be an abstract base class, so subclasses must implement the following
    methods:

    1. `setup_objective`: to setup the loss function
    2. `forward`: to define the forward pass of the model, including how the loss is computed
    """

    MODEL_TYPE = SetTransformerEncoder

    def __init__(self, config: _BaseConfigT):
        super().__init__(config=config, model_type=self.MODEL_TYPE)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        strict: bool = True,
        **update_kwargs,
    ) -> Self:
        """Load a model from a pretrained (or just trained) checkpoint.

        This method handles subclasses that add new trainable parameters that want to start
        from a parent pretrained model. For example, subclassing a `ProteinSetTransformer` model
        and changing the objective from triplet loss to some label classification would introduce
        new trainable layers that would benefit from starting from a pretrained PST.

        This method works with allows interchanging between `SetTransformer` (genomic) and
        `SetTransformerEncoder` (protein) models. The most common case of this would be with
        a pretrained genomic PST that is then finetuned for a protein-level task in which
        the decoder of the `SetTransformer` is not used.

        Args:
            pretrained_model_name_or_path (str | Path): name or file path to the pretrained model
                checkpoint. NOTE: passing model names is not currently supported
            strict (bool, optional): raise a RuntimeError if there are unexpected parameters
                in the checkpoint's state dict. Defaults to True.
            **update_kwargs: additional keyword arguments to update the model config with

        Raises:
            NotImplementedError: Loading models from their names is not implemented yet
        """
        return super().from_pretrained(
            pretrained_model_name_or_path,
            model_type=cls.MODEL_TYPE,
            strict=strict,
            **update_kwargs,
        )

    def _try_load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True):
        try:
            # try loading directly, which should work for pretrained protein PSTs
            # derived from this class
            super()._try_load_state_dict(state_dict, strict=strict)
        except RuntimeError:
            # however, if it fails, it is likely that the ckpt is from a genomic PST
            # so we need to extract the pos/strand embeddings and the SetTransformerEncoder
            # state dict ONLY
            # we do not care about the SetTransformerDecoder state dict!!

            # we just need to rebuild the state dict to only include the relevant layers

            # get the base parameters of the SetTransformerEncoder from the view of a
            # ProteinSetTransformer: ie from PST, the encoder is under the field model.encoder.AAA
            # also get with the positional and strand embeddings
            params_to_extract = {
                f"model.encoder.{name}" for name, _ in self.model.named_parameters()
            }

            # PositionalStrandEmbeddingModuleMixin params
            for embedding_name in ("positional_embedding", "strand_embedding"):
                layer: torch.nn.Module = getattr(self, embedding_name)
                for name, _ in layer.named_parameters():
                    params_to_extract.add(f"{embedding_name}.{name}")

            # from PST, the encoder is under the field model.encoder.AAA
            # but for the protein PST, the expected field is model.AAA
            new_state_dict = {
                name.replace("encoder.", ""): state_dict[name] for name in params_to_extract
            }

            # get all new params
            current_params = {name for name, _ in self.named_parameters()}

            new_params = current_params - new_state_dict.keys()

            # now try to load the state dict
            missing, unexpected = map(set, self.load_state_dict(new_state_dict, strict=False))

            # missing should be equivalent to the new params if loaded correctly
            still_missing = new_params - missing

            if still_missing:
                raise RuntimeError(
                    f"Missing parameters: {still_missing} when loading the state dict"
                )

            if strict and unexpected:
                raise RuntimeError(
                    f"Unexpected parameters: {unexpected} when loading the state dict"
                )

    def forward_step(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        ptr: torch.Tensor,
        batch: OptTensor = None,
        node_mask: OptTensor = None,
        return_attention_weights: bool = False,
    ) -> EdgeAttnOutput:
        # ptr is not used here but needed for the signature due to inheritance
        output: EdgeAttnOutput = self.model(
            x=x,
            edge_index=edge_index,
            batch=batch,
            node_mask=node_mask,
            return_attention_weights=return_attention_weights,
        )

        return output

    # change type annotations
    def databatch_forward(
        self,
        batch: GenomeGraphBatch,
        node_mask: OptTensor = None,
        return_attention_weights: bool = False,
        x: OptTensor = None,
    ) -> EdgeAttnOutput:
        result = super().databatch_forward(
            batch=batch,
            node_mask=node_mask,
            return_attention_weights=return_attention_weights,
            x=x,
        )

        return cast(EdgeAttnOutput, result)

    def predict_step(
        self, batch: GenomeGraphBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> EdgeAttnOutput:
        result = super().predict_step(
            batch=batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx
        )

        return cast(EdgeAttnOutput, result)


# these must have the encoder first
BaseModelTypes = Union[
    type[BaseProteinSetTransformerEncoder],
    type[BaseProteinSetTransformer],
]

BaseModels = Union[
    BaseProteinSetTransformerEncoder,
    BaseProteinSetTransformer,
]
