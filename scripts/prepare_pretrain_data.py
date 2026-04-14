import os
import argparse
import numpy as np

from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm
import truecase
import nltk
nltk.download('punkt_tab')


"""
prepare_data
    Streams a HuggingFace text dataset, tokenizes every document,
    appends an <|endoftext|> boundary token between documents, and
    writes the full token stream to a flat binary file of uint16 values.

    The output file can be memory-mapped at training time (see
    data/CorpusDataset.py), avoiding the need to load the entire
    corpus into RAM.

    Args:
        tokenizer_path: string path to a saved tokenizer JSON
        dataset_name: string HuggingFace dataset identifier
        text_field: string name of the text column in the dataset
        output_path: string path for the output .bin file
        max_tokens: int stop after accumulating this many tokens (0 = no limit)
        chunk_size: int number of documents to accumulate before flushing to disk
"""
def prepare_data(tokenizer_path, dataset, subset, text_field, output_path, max_tokens, chunk_size, do_truecase=False):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()

    # uint16 fits up to 65535 — sufficient for vocab sizes up to ~50k
    assert vocab_size <= 65535, \
        f"vocab size {vocab_size} exceeds uint16 range; use uint32 dtype instead"

    print(f"[prepare_data] tokenizer: '{tokenizer_path}' (vocab size {vocab_size:,})")
    print(f"[prepare_data] dataset:   '{dataset}'")
    print(f"[prepare_data] output:    '{output_path}'")
    if max_tokens > 0:
        print(f"[prepare_data] max tokens: {max_tokens:,}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dataset = load_dataset(dataset, subset, split="train", streaming=True, trust_remote_code=True)

    total_tokens = 0
    chunk = []

    with open(output_path, "wb") as f:
        pbar = tqdm(desc="Tokenizing documents", unit=" docs")
        for example in dataset:
            text = example[text_field]
            if do_truecase:
                text = truecase.get_true_case(text)
            ids = tokenizer.encode(text).ids
            chunk.extend(ids)
            total_tokens += len(ids)
            pbar.update(1)
            pbar.set_postfix(tokens=f"{total_tokens:,}")

            # flush to disk periodically to keep memory usage bounded
            if len(chunk) >= chunk_size:
                np.array(chunk, dtype=np.uint16).tofile(f)
                chunk = []

            if max_tokens > 0 and total_tokens >= max_tokens:
                break

        # flush remaining tokens
        if chunk:
            np.array(chunk, dtype=np.uint16).tofile(f)

        pbar.close()

    file_size_gb = os.path.getsize(output_path) / 1e9
    print(f"[prepare_data] done — {total_tokens:,} tokens written ({file_size_gb:.2f} GB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer.json")
    parser.add_argument("--dataset", type=str, default="HuggingFaceTB/smollm-corpus")
    parser.add_argument("--subset", type=str, default="cosmopedia-v2")
    parser.add_argument("--text_field", type=str, default="text")
    parser.add_argument("--output", type=str, default="data/pretrain/cosmopedia-v2.bin")
    parser.add_argument("--max_tokens", type=int, default=0)
    parser.add_argument("--chunk_size", type=int, default=1_000_000)
    parser.add_argument("--truecase", action="store_true")
    args = parser.parse_args()

    prepare_data(
        tokenizer_path=args.tokenizer,
        dataset=args.dataset,
        subset=args.subset,
        text_field=args.text_field,
        output_path=args.output,
        max_tokens=args.max_tokens,
        chunk_size=args.chunk_size,
        do_truecase=args.truecase,
    )
