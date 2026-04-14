import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import argparse
from tokenizers import Tokenizer
import time
import sys

from models.LanguageTransformer import LanguageTransformer


# dynamically select device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


"""
sample
    Draws a single next-token index from the model's output logits
    using temperature scaling followed by optional top-k and/or
    top-p (nucleus) filtering.

Args:
    logits:      torch.Tensor of shape (vocab_size,) — raw logits for the next token
    temperature: float > 0, scales logit sharpness; 1.0 = unmodified
    top_k:       int or None; if set, keeps only the k highest-probability tokens
    top_p:       float in (0, 1] or None; if set, keeps the smallest set of tokens
                 whose cumulative probability >= top_p (nucleus sampling)

Returns:
    int token index
"""
def sample(logits, temperature=1.0, top_k=None, top_p=None):
    logits = logits / temperature

    if top_k is not None:
        # zero out all logits outside the top-k
        threshold = torch.topk(logits, top_k).values[-1]
        logits[logits < threshold] = float('-inf')

    if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        # remove tokens once cumulative prob exceeds top_p
        sorted_indices_to_remove = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[sorted_indices_to_remove] = float('-inf')
        logits = torch.scatter(logits, 0, sorted_indices, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


"""
generate
    Runs an autoregressive inference loop: encodes the prompt, feeds it
    through the model, samples the next token, appends it to the context,
    and repeats for max_new_tokens steps.  The context is truncated to
    max_context tokens (the model's effective window) if it grows too long.

Args:
    model:          torch.nn.Module — loaded LanguageTransformer model in eval mode
    tokenizer:      tokenizers.Tokenizer — custom BPE tokenizer
    prompt:         str — text prompt to condition generation on
    max_new_tokens: int — number of tokens to generate
    max_context:    int — maximum sequence length fed to the model
    temperature:    float — sampling temperature
    top_k:          int or None — top-k filtering cutoff
    top_p:          float or None — nucleus sampling cutoff

Returns:
    str generated text (prompt + new tokens decoded)
"""
@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=256,
    max_context=1024,
    temperature=1.0,
    top_k=50,
    top_p=0.99,
):
    model.eval()
    model.reset_cache()

    encoded = tokenizer.encode(prompt)
    token_ids = encoded.ids
    print(prompt, end="")

    all_ids = list(token_ids)

    # Prefill: process the full prompt in one pass and populate the KV cache.
    context = torch.tensor([token_ids], dtype=torch.long, device=device)
    logits = model(context, use_cache=True, start_pos=0)
    start_pos = len(token_ids)

    for _ in range(max_new_tokens):
        next_logits = logits[0, -1]
        next_token = sample(next_logits, temperature=temperature, top_k=top_k, top_p=top_p)
        next_word = tokenizer.decode([next_token])
        sys.stdout.write(next_word)
        sys.stdout.flush()
        all_ids.append(next_token)

        if start_pos >= max_context:
            # Sliding-window fallback: rebuild the cache from the last max_context tokens.
            model.reset_cache()
            ctx = torch.tensor([all_ids[-max_context:]], dtype=torch.long, device=device)
            logits = model(ctx, use_cache=True, start_pos=0)
            start_pos = max_context
        else:
            # Decode step: feed only the single new token at its correct position.
            next_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
            logits = model(next_tensor, use_cache=True, start_pos=start_pos)
            start_pos += 1

    return tokenizer.decode(all_ids)


"""
load_model
    Loads a LanguageTransformer checkpoint, reconstructs the model from the saved
    model_config, and returns both the model and the tokenizer.

Args:
    checkpoint_path: str path to the .pth checkpoint file
    tokenizer_path:  str or None; overrides the path stored in the
                     checkpoint (useful when running from a different
                     working directory)

Returns:
    (model, tokenizer) tuple
"""
def load_model(checkpoint_path, tokenizer_path):
    chkpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model_config = chkpt['model_config']

    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()

    model = LanguageTransformer(
        vocab_size=vocab_size,
        embed_dim=model_config['emb_dim'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        is_causal=True
    ).to(device)

    model.load_state_dict(chkpt['model_state_dict'])
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="data/tokenizers/tokenizer.json")
    parser.add_argument("--prompt", default="The meaning of life is")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_context", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    args = parser.parse_args()

    top_k = args.top_k if args.top_k > 0 else None
    top_p = args.top_p if args.top_p < 1.0 else None

    model, tokenizer = load_model(args.checkpoint, args.tokenizer)
    print("")

    output = generate(
        model,
        tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        max_context=args.max_context,
        temperature=args.temperature,
        top_k=top_k,
        top_p=top_p,
    )
    print("")


if __name__ == "__main__":
    main()
