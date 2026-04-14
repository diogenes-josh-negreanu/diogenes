import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader


"""
CorpusDataset
    Memory-mapped dataset over a flat binary token file produced by
    scripts/prepare_data.py. Slices the token stream into fixed-length
    chunks of size chunk_len + 1, where the extra token provides the
    shifted label for next-token prediction.

    Because documents are concatenated end-to-end (separated by
    <|endoftext|> tokens), there is no padding — every position in
    every batch contributes to the loss.
"""
class CorpusDataset(Dataset):
    """
    CorpusDataset.__init__
        Opens the binary token file as a read-only memory map.

        Args:
            path: string path to a .bin file of uint16 token IDs
            chunk_len: int number of input tokens per training example
    """
    def __init__(self, path, chunk_len):
        self.data = np.memmap(path, dtype=np.uint16, mode='r')
        self.chunk_len = chunk_len
        print(f"[CorpusDataset] {len(self.data):,} tokens, {len(self):,} chunks of length {chunk_len}")

    """
    CorpusDataset.__len__
        Returns the number of non-overlapping chunks in the dataset.

        Returns:
            int number of training examples
    """
    def __len__(self):
        # subtract 1 so the last chunk always has a label token
        return (len(self.data) - 1) // self.chunk_len

    """
    CorpusDataset.__getitem__
        Returns a single chunk of chunk_len + 1 tokens as a long tensor.
        In the training loop, input is chunk[:-1] and label is chunk[1:].

        Args:
            idx: int index of the chunk

        Returns:
            torch.Tensor of shape (chunk_len + 1,) and dtype torch.long
    """
    def __getitem__(self, idx):
        start = idx * self.chunk_len
        chunk = self.data[start : start + self.chunk_len + 1].astype(np.int64)
        return torch.from_numpy(chunk)

    """
    CorpusDataset.create_dataloader
        Creates a DataLoader for this dataset.

    Args:
        bs: int batch size
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
            drop_last=True
        )
