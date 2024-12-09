from itertools import chain

import esm
import torch
from torch.utils.data import Dataset

from pst.embed.model import BatchType


class SequenceDataset(Dataset):
    """Sequence dataset that pre-batches input sequences by number of tokens"""

    # TODO: should create custom FastaBatchedDataset to read seqs and remove stop codons
    def __init__(
        self, data: esm.FastaBatchedDataset, alphabet: esm.Alphabet, batch_size: int
    ) -> None:
        """Create sequence dataset object.

        Args:
            data (esm.FastaBatchedDataset): Fasta dataset from
                `esm.FastaBatchedDataset.from_file`
            alphabet (esm.Alphabet): alphabet from esm model
            batch_size (int): size in number of tokens (ie number of residues)
        """
        self.data = data
        # this is in number of tokens, not individual seqs
        self.batch_size = batch_size

        self.batch_converter = alphabet.get_batch_converter()
        self.batch_indices = data.get_batch_indices(batch_size)

    def __len__(self) -> int:
        return len(self.batch_indices)

    def __getitem__(self, idx: int) -> BatchType:
        batch_idx = self.batch_indices[idx]
        batch = [self.data[i] for i in batch_idx]
        return self.batch_converter(batch)

    @staticmethod
    def collate_token_batches(batches: list[BatchType]) -> BatchType:
        """Custom `collate_fn` that reorganizes a batch list of `BatchType` objects
        that are produced by the standard PyTorch `DataLoader` into a flattened
        `BatchType` object.

        Args:
            batches (list[BatchType]): list of batch objects from PyTorch `DataLoader`

        Returns:
            BatchType: batches gets flattened into a single batch
        """
        labels: tuple[list[str], ...]
        seqs: tuple[list[str], ...]
        tokens: tuple[torch.Tensor, ...]
        labels, seqs, tokens = zip(*batches)
        batch_labels = list(chain.from_iterable(labels))
        batch_seqs = list(chain.from_iterable(seqs))
        batch_tokens = torch.vstack(tokens)
        return batch_labels, batch_seqs, batch_tokens
