from itertools import chain

import esm
import torch
from torch.utils.data import Dataset

from pst.embed.model import BatchType
from pst.typing import FilePath


class FastaBatchedDataset(esm.FastaBatchedDataset):
    @classmethod
    def from_file(cls, fasta_file: FilePath):
        sequence_labels: list[str] = []
        sequence_strs: list[str] = []
        cur_seq_label: str | None = None
        buf: list[str] = []

        def _flush_current_seq():
            nonlocal cur_seq_label, buf
            if cur_seq_label is None:
                return
            sequence_labels.append(cur_seq_label)

            # remove stop codons
            sequence = "".join(buf).replace("*", "")
            sequence_strs.append(sequence)
            cur_seq_label = None
            buf = []

        with open(fasta_file, "r") as infile:
            for line_idx, line in enumerate(infile):
                if line.startswith(">"):  # label line
                    _flush_current_seq()

                    # this is the header
                    line = line[1:].strip()
                    if len(line) > 0:
                        cur_seq_label = line
                    else:
                        cur_seq_label = f"seqnum{line_idx:09d}"
                else:  # sequence line
                    buf.append(line.strip())

        _flush_current_seq()

        assert len(set(sequence_labels)) == len(sequence_labels), (
            "Found duplicate sequence labels"
        )

        return cls(sequence_labels, sequence_strs)


class SequenceDataset(Dataset):
    """Sequence dataset that pre-batches input sequences by number of tokens"""

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

        self.seq2order: dict[str, int] = {
            seq: i for i, seq in enumerate(self.data.sequence_labels)
        }

        # this is in number of tokens, not individual seqs
        self.batch_size = batch_size

        self.batch_converter = alphabet.get_batch_converter()
        self.batch_indices = data.get_batch_indices(batch_size)

    @classmethod
    def from_file(
        cls, fasta_file: FilePath, alphabet: esm.Alphabet, batch_size: int
    ):
        return cls(
            data=FastaBatchedDataset.from_file(fasta_file),
            alphabet=alphabet,
            batch_size=batch_size,
        )

    @property
    def num_sequences(self) -> int:
        return len(self.data.sequence_labels)

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
