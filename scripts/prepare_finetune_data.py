import os
import re
import random
import argparse
import numpy as np

from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm


def remove_citations(text):
    """Remove inline citation references like [1], [2], [1,2], [1-3], etc."""
    return re.sub(r'\[\d+(?:[,\-]\d+)*\]', '', text)


"""
tokenize_example
    Formats a single question/answer pair into ChatML, tokenizes it
    segment-by-segment so we can build an exact loss mask without
    scanning the full token stream. Because <|im_start|> and <|im_end|>
    are special tokens they act as hard BPE boundaries, so tokenizing
    the content between them in isolation gives the same IDs as
    tokenizing the full string.

    Loss mask:
        0 — user turn (not trained on)
        1 — assistant response + trailing <|im_end|> (trained on)

    Format produced:
        <|im_start|>system
        {context}<|im_end|>
        <|im_start|>user
        {question}<|im_end|>
        <|im_start|>assistant
        {answer}<|im_end|>

Args:
    tokenizer: Tokenizer loaded from JSON
    question: str user message text
    answer: str assistant response text

Returns:
    tuple (ids: list[int], mask: list[int])
"""
def tokenize_example(tokenizer, question, answer):
    # fixed structural segments — tokenized once outside and reused per call
    user_start   = tokenizer.encode("<|im_start|>user\n").ids
    turn_end     = tokenizer.encode("<|im_end|>\n").ids
    asst_start   = tokenizer.encode("<|im_start|>assistant\n").ids

    q_ids = tokenizer.encode(question).ids
    a_ids = tokenizer.encode(answer).ids

    ids = user_start + q_ids + turn_end + asst_start + a_ids + turn_end
    mask = (
        [0] * len(user_start) +
        [0] * len(q_ids) +
        [0] * len(turn_end) +
        [0] * len(asst_start) +
        [1] * len(a_ids) +
        [1] * len(turn_end)   # include the closing <|im_end|> so model learns to stop
    )
    return ids, mask


"""
tokenize_chat_history
    Formats a multi-turn chat history (list of role/content dicts) into
    ChatML tokens with a per-token loss mask. Handles system, user, and
    assistant roles in any order, matching the format used at inference
    in chat.py:build_prompt.

    Loss mask:
        0 — system and user turns (not trained on)
        1 — assistant content + trailing <|im_end|> (trained on)

    Format produced per message:
        <|im_start|>{role}
        {content}<|im_end|>

Args:
    tokenizer: Tokenizer loaded from JSON
    messages:  list[dict]  each dict must have 'role' and 'content' keys

Returns:
    tuple (ids: list[int], mask: list[int])
    Returns ([], []) if the history contains no assistant turn.
"""
def tokenize_chat_history(tokenizer, messages):
    turn_end = tokenizer.encode("<|im_end|>\n").ids

    ids  = []
    mask = []
    has_assistant = False

    for msg in messages:
        role    = msg.get('role', '')
        content = msg.get('content', '')

        if not isinstance(content, str) or not content.strip():
            continue

        header      = tokenizer.encode(f"<|im_start|>{role}\n").ids
        content_ids = tokenizer.encode(content).ids

        if role == 'assistant':
            has_assistant = True
            ids  += header + content_ids + turn_end
            mask += [0] * len(header) + [1] * len(content_ids) + [1] * len(turn_end)
        else:
            ids  += header + content_ids + turn_end
            mask += [0] * (len(header) + len(content_ids) + len(turn_end))

    if not has_assistant:
        return [], []

    return ids, mask


"""
prepare_sft_data
    Tokenizes a HuggingFace dataset into two flat binary files:
        {output}          — uint16 token IDs (same layout as pretraining .bin)
        {output_mask}     — uint8  loss mask  (1 = compute loss, 0 = ignore)

    Examples are packed end-to-end with no padding, so the files can be
    memory-mapped by SFTDataset exactly like CorpusDataset reads its .bin.

Args:
    tokenizer_path:  str  path to tokenizer JSON
    dataset_name:    str  HuggingFace dataset identifier
    output_path:     str  path for the token IDs .bin file
    split:           str  dataset split to use (default "train")
    max_examples:    int  stop after this many examples (0 = full dataset)
    chunk_size:      int  flush to disk after accumulating this many tokens
"""
_DEFAULT_SYSTEM_PROMPTS = [
    "You are a helpful AI assistant.",
    "You are a knowledgeable and helpful assistant. Answer clearly and concisely.",
    "You are a helpful, accurate, and thoughtful assistant.",
    "You are an intelligent assistant. Help the user with their questions and tasks.",
    "You are a friendly and knowledgeable AI. Provide clear, accurate responses.",
    "You are a helpful assistant that provides informative and concise answers.",
    "You are a smart AI assistant. Be helpful, harmless, and honest.",
]


def prepare_sft_data(
    tokenizer_path,
    dataset_name,
    output_path,
    split="train",
    max_examples=0,
    chunk_size=1_000_000,
    chat_field='messages',
    inject_system=True,
):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()

    assert vocab_size <= 65535, (
        f"vocab size {vocab_size} exceeds uint16 range; use uint32 dtype instead"
    )

    # derive mask path: data/tokenized/sft.bin -> data/tokenized/sft_mask.bin
    base, ext = os.path.splitext(output_path)
    mask_path = base + "_mask" + ext

    print(f"[prepare_sft_data] tokenizer:      '{tokenizer_path}' (vocab {vocab_size:,})")
    print(f"[prepare_sft_data] dataset:         '{dataset_name}' split='{split}'")
    print(f"[prepare_sft_data] tokens output:   '{output_path}'")
    print(f"[prepare_sft_data] mask output:     '{mask_path}'")
    print(f"[prepare_sft_data] inject_system:   {inject_system}")
    if max_examples > 0:
        print(f"[prepare_sft_data] max examples:   {max_examples:,}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dataset = load_dataset(dataset_name, 'all', split=split, streaming=True, trust_remote_code=True)

    total_tokens = 0
    total_assistant_tokens = 0
    skipped = 0
    token_chunk = []
    mask_chunk  = []

    with open(output_path, "wb") as ftok, open(mask_path, "wb") as fmask:
        pbar = tqdm(desc="Tokenizing examples", unit=" ex")

        for i, example in enumerate(dataset):
            if max_examples > 0 and i >= max_examples:
                break
            
            messages = example.get(chat_field)
            if not isinstance(messages, list) or len(messages) == 0:
                skipped += 1
                continue

            if inject_system and not any(m.get('role') == 'system' for m in messages):
                system_prompt = random.choice(_DEFAULT_SYSTEM_PROMPTS)
                messages = [{"role": "system", "content": system_prompt}] + messages

            ids, mask = tokenize_chat_history(tokenizer, messages)
            if not ids:
                skipped += 1
                continue

            token_chunk.extend(ids)
            mask_chunk.extend(mask)
            total_tokens += len(ids)
            total_assistant_tokens += sum(mask)

            pbar.update(1)
            pbar.set_postfix(tokens=f"{total_tokens:,}")

            # flush to disk periodically to keep memory usage bounded
            if len(token_chunk) >= chunk_size:
                np.array(token_chunk, dtype=np.uint16).tofile(ftok)
                np.array(mask_chunk,  dtype=np.uint8).tofile(fmask)
                token_chunk = []
                mask_chunk  = []

        # flush remainder
        if token_chunk:
            np.array(token_chunk, dtype=np.uint16).tofile(ftok)
            np.array(mask_chunk,  dtype=np.uint8).tofile(fmask)

        pbar.close()

    tok_size_mb  = os.path.getsize(output_path) / 1e6
    mask_size_mb = os.path.getsize(mask_path)   / 1e6
    print(f"[prepare_sft_data] done")
    print(f"  total tokens:     {total_tokens:,}")
    print(f"  assistant tokens: {total_assistant_tokens:,} "
          f"({100 * total_assistant_tokens / max(total_tokens, 1):.1f}% of stream trained on)")
    if skipped:
        print(f"  skipped:          {skipped:,} (empty, missing field, or no assistant turn)")
    print(f"  tokens file:      {tok_size_mb:.1f} MB")
    print(f"  mask file:        {mask_size_mb:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer.json")
    parser.add_argument("--dataset", type=str, default="HuggingFaceTB/smoltalk")
    parser.add_argument("--output", type=str, default="data/tokenized/sft/smoltalk.bin")
    parser.add_argument("--chat_field", type=str, default="messages")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument("--chunk_size", type=int, default=1_000_000)
    parser.add_argument("--no_inject_system", action="store_true",
                        help="disable default system prompt injection for examples without one")
    args = parser.parse_args()

    prepare_sft_data(
        tokenizer_path=args.tokenizer,
        dataset_name=args.dataset,
        output_path=args.output,
        split=args.split,
        max_examples=args.max_examples,
        chunk_size=args.chunk_size,
        chat_field=args.chat_field,
        inject_system=not args.no_inject_system,
    )
