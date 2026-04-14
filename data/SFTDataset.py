import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader


"""
SFTDataset
    Memory-mapped dataset over the two binary files produced by
    scripts/prepare_sft_data.py:
        tokens_path — flat uint16 token IDs (identical layout to CorpusDataset)
        mask_path   — flat uint8 loss mask (1 = compute loss, 0 = ignore)

    For each chunk the dataset returns (inp, labels) where labels has -100
    at every position where mask == 0. CrossEntropyLoss ignores -100 by
    default, so only assistant-response tokens contribute to the loss.
"""
class SFTDataset(Dataset):
    """
    SFTDataset.__init__
        Opens both binary files as read-only memory maps.

    Args:
        tokens_path: str path to the uint16 token IDs .bin file
        mask_path:   str path to the uint8 loss mask .bin file
        chunk_len:   int number of input tokens per training example
    """
    def __init__(self, tokens_path, mask_path, chunk_len):
        self.tokens    = np.memmap(tokens_path, dtype=np.uint16, mode='r')
        self.mask      = np.memmap(mask_path,   dtype=np.uint8,  mode='r')
        self.chunk_len = chunk_len

        assert len(self.tokens) == len(self.mask), (
            f"tokens ({len(self.tokens):,}) and mask ({len(self.mask):,}) length mismatch"
        )
        print(f"[SFTDataset] {len(self.tokens):,} tokens, {len(self):,} chunks of length {chunk_len}")

    """
    SFTDataset.__len__
        Returns the number of non-overlapping chunks.
    """
    def __len__(self):
        return (len(self.tokens) - 1) // self.chunk_len

    """
    SFTDataset.__getitem__
        Returns (inp, labels) for one chunk. Labels at positions where
        mask == 0 are set to -100 so they are excluded from the loss.

    Args:
        idx: int chunk index

    Returns:
        inp:    torch.Tensor shape (chunk_len,)   dtype long
        labels: torch.Tensor shape (chunk_len,)   dtype long  (-100 at non-assistant positions)
    """
    def __getitem__(self, idx):
        start = idx * self.chunk_len
        end   = start + self.chunk_len + 1

        tokens = self.tokens[start:end].astype(np.int64)
        mask   = self.mask[start:end].astype(np.int64)

        tokens_t = torch.from_numpy(tokens)
        mask_t   = torch.from_numpy(mask)

        inp    = tokens_t[:-1]
        labels = tokens_t[1:].clone()
        labels[mask_t[1:] == 0] = -100   # ignore all non-assistant positions

        return inp, labels

    """
    SFTDataset.create_dataloader
        Creates a DataLoader for this dataset.

    Args:
        bs:          int batch size
        num_workers: int number of DataLoader worker processes

    Returns:
        torch.utils.data.DataLoader
    """
    def create_dataloader(self, bs, num_workers=4):
        return DataLoader(
            self,
            batch_size=bs,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
