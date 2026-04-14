import os
import argparse

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


"""
text_iterator
    Streams text examples from a HuggingFace dataset, yielding
    one string at a time. Stops after max_examples documents.

    Args:
        dataset_name: string HuggingFace dataset identifier
        text_field: string name of the text column
        max_examples: int number of documents to use for training
"""
def text_iterator(dataset, subset, text_field, max_examples):
    dataset = load_dataset(dataset, subset, split="train", streaming=True, trust_remote_code=True)
    for i, example in enumerate(dataset):
        if i >= max_examples:
            break
        yield example[text_field]


"""
train_tokenizer
    Trains a BPE tokenizer on a subset of a HuggingFace
    text dataset. Saves the trained tokenizer to a JSON file.

    Special tokens:
        <|pad|>       (ID 0) padding
        <|im_start|>  (ID 1) start of a chat turn (user or assistant)
        <|im_end|>    (ID 2) end of a chat turn — generation stop token

    Args:
        dataset_name: string HuggingFace dataset identifier
        text_field: string name of the text column in the dataset
        vocab_size: int target vocabulary size
        max_examples: int number of documents to sample for training
        output_path: string path to save the tokenizer JSON
"""
def train_tokenizer(dataset, subset, text_field, vocab_size, max_examples, output_path):
    print(f"[train_tokenizer] training on {max_examples:,} examples from '{dataset}'")
    print(f"[train_tokenizer] target vocab size: {vocab_size:,}")

    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    # chatML special tokens
    special_tokens = ["<|pad|>", "<|im_start|>", "<|im_end|>"]
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True
    )

    tokenizer.train_from_iterator(
        text_iterator(dataset, subset, text_field, max_examples),
        trainer=trainer
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tokenizer.save(output_path)

    print(f"[train_tokenizer] saved to '{output_path}'")
    print(f"[train_tokenizer] final vocab size: {tokenizer.get_vocab_size():,}")
    for tok in special_tokens:
        print(f"  {tok!r:20s} -> ID {tokenizer.token_to_id(tok)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="HuggingFaceTB/smollm-corpus")
    parser.add_argument("--subset", type=str, default="cosmopedia-v2")
    parser.add_argument("--text_field", type=str, default="text")
    parser.add_argument("--vocab_size", type=int, default=50256)
    parser.add_argument("--max_examples", type=int, default=500_000)
    parser.add_argument("--output", type=str, default="data/beta_tokenizer.json")
    args = parser.parse_args()

    train_tokenizer(
        dataset=args.dataset,
        subset=args.subset,
        text_field=args.text_field,
        vocab_size=args.vocab_size,
        max_examples=args.max_examples,
        output_path=args.output
    )
