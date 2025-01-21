from typing import Any, cast

import torch

from pst.nn.base import (
    _FIXED_POINTSWAP_RATE,
    BaseProteinSetTransformer,
    BaseProteinSetTransformerEncoder,
)
from pst.nn.config import (
    GenomeTripletLossModelConfig,
    MaskedLanguageModelingConfig,
    ProteinTripletLossModelConfig,
)
from pst.nn.utils.distance import pairwise_euclidean_distance, stacked_batch_chamfer_distance
from pst.nn.utils.loss import (
    AugmentedWeightedTripletLoss,
    MaskedLanguageModelingLoss,
    WeightedTripletLoss,
)
from pst.nn.utils.mask import mask_batch
from pst.nn.utils.sampling import negative_sampling, point_swap_sampling, positive_sampling
from pst.typing import GenomeGraphBatch


class ProteinSetTransformer(BaseProteinSetTransformer[GenomeTripletLossModelConfig]):
    # NOTE: updated as new pretrained models are added
    PRETRAINED_MODEL_NAMES = {"vpst-small", "vpst-large"}

    def __init__(self, config: GenomeTripletLossModelConfig):
        # this only needs to be defined for the config type hint
        super().__init__(config=config)

    def setup_objective(self, margin: float, **kwargs) -> AugmentedWeightedTripletLoss:
        return AugmentedWeightedTripletLoss(margin=margin)

    def forward(self, batch: GenomeGraphBatch, augment_data: bool = True) -> torch.Tensor:
        """Forward pass using Point Swap augmentation (during training only) with a triplet loss function."""

        # adding positional and strand embeddings lead to those dominating the plm signal
        # we can concatenate them here, then use a linear layer to project down back to
        # the original feature dim and force the model to directly learn which of these
        # are most important

        # NOTE: we do not adjust the original data at batch.x
        # this lets the augmented data adjust the positional and strand embeddings
        # independently of the original data
        x, positional_embed, strand_embed = self.internal_embeddings(batch)

        # calculate chamfer distance only based on the plm embeddings
        # want to maximize that signal over strand and positional embeddings
        setwise_dist, item_flow = stacked_batch_chamfer_distance(batch=batch.x, ptr=batch.ptr)
        setwise_dist_std = setwise_dist.std()

        #### REAL DATA ####
        # positive mining
        pos_idx = positive_sampling(setwise_dist)

        # forward pass
        y_anchor, _ = self.databatch_forward(
            batch=batch,
            return_attention_weights=False,
            x=x,
        )

        # negative sampling
        neg_idx, neg_weights = negative_sampling(
            input_space_pairwise_dist=setwise_dist,
            output_embed_X=y_anchor,
            input_space_dist_std=setwise_dist_std,
            pos_idx=pos_idx,
            scale=self.config.loss.sample_scale,
            no_negatives_mode=self.config.loss.no_negatives_mode,
        )

        y_pos = y_anchor[pos_idx]
        y_neg = y_anchor[neg_idx]

        if augment_data:
            y_aug_pos, y_aug_neg, aug_neg_weights = self._augmented_forward_step(
                batch=batch,
                pos_idx=pos_idx,
                y_anchor=y_anchor,
                item_flow=item_flow,
                positional_embed=positional_embed,
            )
        else:
            y_aug_pos = None
            y_aug_neg = None
            aug_neg_weights = None

        loss: torch.Tensor = self.criterion(
            y_self=y_anchor,
            y_pos=y_pos,
            y_neg=y_neg,
            neg_weights=neg_weights,
            class_weights=batch.weight,  # type: ignore
            y_aug_pos=y_aug_pos,
            y_aug_neg=y_aug_neg,
            aug_neg_weights=aug_neg_weights,
        )

        return loss

    def _augmented_forward_step(
        self,
        batch: GenomeGraphBatch,
        pos_idx: torch.Tensor,
        y_anchor: torch.Tensor,
        item_flow: torch.Tensor,
        positional_embed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        augmented_batch, aug_idx = point_swap_sampling(
            batch=batch.x,
            pos_idx=pos_idx,
            item_flow=item_flow,
            sizes=batch.num_proteins,
            sample_rate=self.config.augmentation.sample_rate,
        )

        # let strand use original strand for each ptn
        strand = batch.strand[aug_idx]
        strand_embed = self.strand_embedding(strand)

        # however instead of changing the positional idx, just keep the same
        # this is basically attempting to mirror same protein encoded in a diff position
        x_aug = self.concatenate_embeddings(
            x=augmented_batch,
            positional_embed=positional_embed,
            strand_embed=strand_embed,
        )

        y_aug_pos, _ = self.databatch_forward(
            batch=batch,
            return_attention_weights=False,
            x=x_aug,
        )

        # NOTE: computing chamfer distance without positional or strand info
        setdist_real_aug, _ = stacked_batch_chamfer_distance(
            batch=batch.x, ptr=batch.ptr, other=augmented_batch
        )

        aug_neg_idx, aug_neg_weights = negative_sampling(
            input_space_pairwise_dist=setdist_real_aug,
            output_embed_X=y_anchor,
            output_embed_Y=y_aug_pos,
            scale=self.config.loss.sample_scale,
            no_negatives_mode=self.config.loss.no_negatives_mode,
        )

        y_aug_neg = y_aug_pos[aug_neg_idx]
        return y_aug_pos, y_aug_neg, aug_neg_weights

    @staticmethod
    def _adjust_checkpoint_inplace(ckpt: dict[str, Any]):
        # fix the sample rate issue first
        # there was an old error when computing the pointswap rate to be 1 - expected
        # the code has been changed (see commit 82b0698)
        # however, old checkpoints will have the previous value, which needs to be adjusted
        hparams = ckpt["hyper_parameters"]
        fixed_pointswap_rate = hparams.pop(_FIXED_POINTSWAP_RATE, None)

        # needed for backwards compatibility
        # the model config is stored in the "config" key
        # in future versions
        if "config" in hparams:
            hparams = hparams["config"]

        if fixed_pointswap_rate is None:
            # key not present = old model
            # so need to adjust the sample rate
            curr_rate = hparams["augmentation"]["sample_rate"]
            hparams["augmentation"]["sample_rate"] = 1.0 - curr_rate

        # move sample_scale and no_negatives_mode to the loss field
        # if they are part of the augmentation field
        for hparam_name in ("sample_scale", "no_negatives_mode"):
            if hparam_name in hparams["augmentation"]:
                hparams["loss"][hparam_name] = hparams["augmentation"].pop(hparam_name)

    ### need to overwrite these methods to handle when data augmentaton should occur
    def training_step(self, train_batch: GenomeGraphBatch, batch_idx: int):
        # need to add augment_data=True to the forward pass
        return super().training_step(train_batch, batch_idx, augment_data=True)

    def validation_step(self, val_batch: GenomeGraphBatch, batch_idx: int):
        # no data aug for validation or testing
        return super().validation_step(val_batch, batch_idx, augment_data=False)

    def test_step(self, test_batch: GenomeGraphBatch, batch_idx: int):
        # no data aug for validation or testing
        return super().test_step(test_batch, batch_idx, augment_data=False)


######### Protein-level models #########


class ProteinSetTransformerEncoder(BaseProteinSetTransformerEncoder[ProteinTripletLossModelConfig]):
    def __init__(self, config: ProteinTripletLossModelConfig):
        super().__init__(config=config)

    def setup_objective(self, margin: float, **kwargs) -> WeightedTripletLoss:
        return WeightedTripletLoss(margin=margin)

    def forward(self, batch: GenomeGraphBatch) -> torch.Tensor:
        # adding positional and strand embeddings lead to those dominating the plm signal
        # so we concatenate them here

        # NOTE: we do not adjust the original data at batch.x since we need that
        # for triplet sampling
        x, _, _ = self.internal_embeddings(batch)

        # calculate distances only based on the plm embeddings
        # want to maximize that signal over strand and positional embeddings
        all_pairwise_dist = pairwise_euclidean_distance(batch.x)
        all_pairwise_dist_std = all_pairwise_dist.std()

        ### positive sampling -
        # happens in input pLM embedding space
        # need to set the diagonal to inf to avoid selecting the same protein as the positive example
        pos_idx = positive_sampling(all_pairwise_dist)

        ### semihard negative sampling -
        # choose the negative example that is closest to positive
        # and farther from the anchor than the positive example
        # NOTE: happens in PST contextualized protein embedding space, ie negative examples
        # are chosen dynamically as model updates

        # forward pass -> ctx ptn [P, D]
        y_anchor, _, _ = self.databatch_forward(
            batch=batch,
            return_attention_weights=False,
            x=x,
        )

        neg_idx, neg_weights = negative_sampling(
            input_space_pairwise_dist=all_pairwise_dist,
            output_embed_X=y_anchor,
            input_space_dist_std=all_pairwise_dist_std,
            pos_idx=pos_idx,
            scale=self.config.loss.sample_scale,
            no_negatives_mode=self.config.loss.no_negatives_mode,
        )

        y_pos = y_anchor[pos_idx]
        y_neg = y_anchor[neg_idx]

        loss: torch.Tensor = self.criterion(
            y_self=y_anchor,
            y_pos=y_pos,
            y_neg=y_neg,
            weights=neg_weights,
            class_weights=None,
            reduce=True,
        )

        return loss

    @staticmethod
    def _adjust_checkpoint_inplace(ckpt: dict[str, Any]):
        # move sample_scale and no_negatives_mode to the loss field
        # if they are part of the augmentation field
        hparams = ckpt["hyper_parameters"]

        if "config" in hparams:
            hparams = hparams["config"]

        if "augmentation" in hparams and hparams["augmentation"]:
            for hparam_name in ("sample_scale", "no_negatives_mode"):
                if hparam_name in hparams["augmentation"]:
                    hparams["loss"][hparam_name] = hparams["augmentation"].pop(hparam_name)


class MLMProteinSetTransformer(BaseProteinSetTransformerEncoder[MaskedLanguageModelingConfig]):
    # inherit from Encoder class since we don't actually need the decoder
    # but to get genome representations, we can just average

    def __init__(self, config: MaskedLanguageModelingConfig):
        super().__init__(config=config)

        # TODO: this could technically be a typevar in the base class
        # not sure it matters that much since it should only be used to call the loss fn
        self.criterion = cast(MaskedLanguageModelingLoss, self.criterion)

    def setup_objective(self, masking_rate: float, **kwargs) -> MaskedLanguageModelingLoss:
        # dont actually need masking rate for the loss since that is handled by data processing
        return MaskedLanguageModelingLoss()

    def forward(self, batch: GenomeGraphBatch):
        # this needs to be done BEFORE MASKING
        # get a positive example -> idea is that multiple diff proteins (perhaps of the same family)
        # could occupy the same context

        # theoretically, this could just only be computed with the masked nodes
        # but idk if that would bias the std calc for what is a typical dist
        node_dist = pairwise_euclidean_distance(batch.x)

        # use to weight good/bad choices of positive examples
        node_dist_std = node_dist.std()

        pos_idx = positive_sampling(node_dist)
        pos_dist = node_dist.gather(dim=-1, index=pos_idx.unsqueeze(-1)).squeeze()

        # This wrapper fn mirrors traditional mlm training where the tokens to-be-masked
        # are already masked before the model sees the data. This fn does several things:
        # 1. Compute a node mask stored in batch.node_mask
        # 2. Extract the protein embeddings to-be-masked for the MLM loss fn. These are stored under batch.label
        # 3. Set's the masked protein embeddings to zero in batch.x
        # NOTE: The masked protein embeddings STILL need the positional and strand embeddings, so we add them after this
        # It should be ok for the masked embeddings to have the gradient attached with the positional and strand embeddings....
        # it's not like that's really the mystery here, it's the protein identity that's the mystery
        # more importantly if the pos/strand embeddings were masked, the learned embeddings
        # for each might suffer since we're trying to learn them twice and independently
        # the trivial solution would be for the pos/strand embeddings to all be 0s (ie masked)
        batch = mask_batch(batch, masking_rate=self.config.loss.masking_rate)

        # these are the TARGETS
        masked_embeddings, *_ = self.masked_embeddings(batch)

        # concatenate positional and strand embeddings
        # the masked embeddings should be [0.0, ..., 0.0, POS_EMBS, STRAND_EMBS]
        x, _, _ = self.internal_embeddings(batch)

        # forward pass
        # y shape: [num_proteins, hidden_dim]
        y, *_ = self.databatch_forward(
            batch=batch,
            node_mask=batch.node_mask,
            return_attention_weights=False,
            x=x,
        )

        y_pos = y[pos_idx]
        y_pos_weight_denom = 2 * (node_dist_std * self.config.loss.sample_scale) ** 2
        y_pos_weight = torch.exp(-pos_dist / y_pos_weight_denom)

        loss = self.criterion(
            y_pred=y,
            masked_embeddings=masked_embeddings,
            node_mask=batch.node_mask,
            y_pos=y_pos,
            y_pos_weight=y_pos_weight,
        )

        return loss
