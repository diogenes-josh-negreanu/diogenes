import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import pandas as pd
import random
import spacy

from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from datasets import load_dataset
from transformers import AutoTokenizer
from staticvectors import StaticVectors
from collections import Counter
from tqdm import tqdm

# dynamically select device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


"""
DiffuseVocabulary
    Builds a vocabulary from a text corpus given
    a tokenizer. Adds special characters for masking
    and training.
"""
class DiffuseVocabulary:
    """
    DiffuseVocabulary.__init__
        Internally stores tokenizer, builds vocabulary
        from provided corpus. Defines word2idx and idx2word
        dictionaries for quick lookup.
    
    Args:
        corpus: list of strings (training examples)
        tokenizer: text tokenizer
    """
    def __init__(self, corpus, tokenizer):
        self.tokenizer = tokenizer
        self.word2idx, self.idx2word = self.build_vocab(corpus)

    """
    DiffuseVocabulary.__len__
        Provides number of distinct tokens in vocabulary.
    
    Returns:
        int length of vocabulary
    """
    def __len__(self):
        return len(self.word2idx)
  
    """
    DiffuseVocabulary.text2idx
        Provides quick lookup for encoding a string
        into its respective tokens.

    Args:
        text: string to tokenize
    
    Returns:
        list of int tokens
    """
    def text2idx(self, text):
        tokens = [str(x).strip().lower() for x in self.tokenizer(text)]
        return [self.word2idx[t] if t in self.word2idx.keys() else self.word2idx['<?>'] for t in tokens]

    """
    DiffuseVocabulary.idx2text
        Provides quick lookup for decoding a list of
        tokens into its respective string.
    
    Args:
        idxs: list of int tokens
    
    Returns:
        string of detokenized words
    """
    def idx2text(self, idxs):
        return " ".join([self.idx2word[i] if i in self.idx2word.keys() else '<?>' for i in idxs])


    """
    DiffuseVocabulary.build_vocab
        Iterates through text corpus, tokenizes data.
        Constructs lookup dictionaries for conversion
        between text and tokens. Adds special tokens reserved
        (0, <p>), (1, <s>), (2, </s>), (3, <?>), (4, <m>).
    
    Args:
        corpus: list of strings (training examples)
    
    Returns:
        dictionary to convert text to tokens
        dictionary to convert tokens to text
    """
    def build_vocab(self, corpus):
        cntr = Counter()
        for datapoint in tqdm(corpus):
            cntr.update( [str(x).strip().lower() for x in self.tokenizer(datapoint)] )

        tokens = [t for t,c in cntr.items() if c >= 30]
        word2idx = {t:i+5 for i,t in enumerate(tokens)}
        idx2word = {i+5:t for i,t in enumerate(tokens)}
        
        # padding token
        word2idx['<p>'] = 0
        idx2word[0] = '<p>'

        # start of sequence token
        word2idx['<s>'] = 1
        idx2word[1] = '<s>'

        # end of sequence token
        word2idx['</s>'] = 2
        idx2word[2] = '</s>'

        # unkown word token
        word2idx['<?>'] = 3
        idx2word[3] = '<?>'

        # mask token
        word2idx['<m>'] = 4
        idx2word[4] = '<m>'
        
        return word2idx, idx2word


"""
DiffuseLanguageDataset
    Custom dataset class for storing poetry dataset
    and custom dataloader.
"""
class DiffuseLanguageDataset(Dataset):
    def __init__(self, dataset_name, max_examples, max_len, bs):
        """
        DiffuseLanguageDataset.__init__
            Creates bert tokenizer, tokenizes huggingface language dataset.
            Creates word2vec embeddings for tokenizer vocabulary. Creates
            custom dataloader.
        
        Args:
            dataset_name: string huggingface dataset name
            max_examples: int max number of training examples to sample
            max_len: max length of each example
            bs: int batch size
        """
        dataset = load_dataset(dataset_name, split='train')
        self.data = [x["text"] for x in random.sample(list(dataset), max_examples)]
        print("[dataset] loaded")

        self.tokenizer = spacy.load('en_core_web_sm').tokenizer
        self.vocab = DiffuseVocabulary(self.data, self.tokenizer)
        print("[dataset] tokenized")

        self.max_len = max_len
        self.bs = bs


    """
    DiffuseLanguageDataset.__len__
        Provides length of dataset.
    
    Returns:
        int number of training examples
    """
    def __len__(self):
        return len(self.data)


    """
    DiffuseLanguageDataset.__getitem__
        Returns a single training example pertaining
        to a given index.

    Returns:
        torch.Tensor 
    """
    def __getitem__(self, idx):
        x = self.vocab.text2idx(self.data[idx])
        l = min(self.max_len, len(x))
        numeralized = x[:l]+[self.vocab.word2idx['</s>']]
        return torch.tensor(numeralized)


    """
    DiffuseLanguageDataset.pad_collate
        Pads training examples with token 0 (<p>).
    
    Returns:
        torch.Tensor of padded batch
    """
    @staticmethod
    def pad_collate(batch):
        xx_pad = pad_sequence(batch, batch_first=True, padding_value=0)
        return xx_pad


    """
    VocabularyDataset.create_dataloader
        Creates a dataloader for internal language
        dataset.
    
    Returns:
        torch.utils.data.DataLoader for custom dataset
    """
    def create_dataloader(self):
        return DataLoader(self, batch_size=self.bs, num_workers=4, shuffle=True, collate_fn=self.pad_collate, drop_last=True)