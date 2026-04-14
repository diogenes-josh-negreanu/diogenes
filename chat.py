import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import re
import torch
import argparse
from tokenizers import Tokenizer
import time
import sys


# ── ANSI style helpers ────────────────────────────────────────────────────────

class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"


def _tw():
    """Return usable terminal width, capped at 25."""
    try:
        return min(os.get_terminal_size().columns, 50)
    except OSError:
        return 50


def _sep(newlines=(1, 1)):
    """Print a thin horizontal rule."""
    w = _tw()
    pre  = "\n" * newlines[0]
    post = "\n" * newlines[1]
    sys.stdout.write(f"{pre}  {C.DIM}{'─' * (w - 4)}{C.RESET}{post}")
    sys.stdout.flush()


def _rl(s):
    """Wrap ANSI escapes so readline doesn't mis-count prompt width."""
    return re.sub(r'(\033\[[0-9;]*m)', r'\001\1\002', s)



from models.GPT import GPT
from models.utils import add_lora, freeze_base_model


# dynamically select device
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.empty_cache()
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
def sample(logits, temperature, top_k, top_p):
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
    model:          torch.nn.Module — loaded GPT model in eval mode
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
    max_new_tokens=100,
    max_context=128,
    temperature=1.0,
    top_k=40,
    top_p=0.9,
    text_color="",
    system_token_len=0,
):
    model.eval()

    encoded = tokenizer.encode(prompt)
    token_ids = encoded.ids

    context = torch.tensor([token_ids], dtype=torch.long, device=device)
    new_token_ids = []
    first = True

    for _ in range(max_new_tokens):
        # Truncate to max_context, but always pin the system-prompt prefix so
        # it is never evicted from the context window.
        if system_token_len > 0 and context.shape[1] > max_context:
            sys_part  = context[:, :system_token_len]
            conv_part = context[:, system_token_len:]
            conv_part = conv_part[:, -(max_context - system_token_len):]
            ctx = torch.cat([sys_part, conv_part], dim=1)
        else:
            ctx = context[:, -max_context:]

        logits = model(ctx)
        next_logits = logits[0, -1]

        next_token = sample(next_logits, temperature=temperature, top_k=top_k, top_p=top_p)
        if next_token == 3:  # <|im_end|>
            break
        next_word = tokenizer.decode([next_token])
        if first:
            # clear the "thinking…" placeholder, then set stream colour
            sys.stdout.write(f"\r\033[K  {C.BOLD}Assistant{C.RESET}  {C.DIM}:{C.RESET}  ")
            if text_color:
                sys.stdout.write(text_color)
            first = False
        sys.stdout.write(next_word)
        sys.stdout.flush()
        new_token_ids.append(next_token)
        context = torch.cat([context, torch.tensor([[next_token]], device=device)], dim=1)

    if text_color:
        sys.stdout.write(C.RESET)
        sys.stdout.flush()

    return tokenizer.decode(new_token_ids)


"""
load_model
    Loads a GPT checkpoint, reconstructs the model from the saved
    model_config, and returns both the model and the tokenizer.

Args:
    checkpoint_path: str path to the .pth checkpoint file
    tokenizer_path:  str or None; overrides the path stored in the
                     checkpoint (useful when running from a different
                     working directory)

Returns:
    (model, tokenizer) tuple
"""
def load_model(model_checkpoint_path, tokenizer_path=None):
    chkpt = torch.load(model_checkpoint_path, weights_only=False, map_location=device)
    model_config = chkpt['model_config']

    tok_path = tokenizer_path or chkpt.get('tokenizer_path', 'data/tokenizer.json')
    tokenizer = Tokenizer.from_file(tok_path)
    vocab_size = tokenizer.get_vocab_size()

    model = GPT(
        vocab_size=vocab_size,
        embed_dim=model_config['emb_dim'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        is_causal=True,
    ).to(device)
    model.load_state_dict(chkpt['model_state_dict'])

    return model, tokenizer


def build_prompt(messages, system=None):
    parts = []
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>\n")
    for msg in messages:
        parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def main():
    diogenes_system = "You are Diogenes the Cynic, an ancient Greek philosopher. You are knowledgeable in all domains, and must answer the following questions from the user with short and concise responses."
    default_system = "You are a helpful and friendly AI assistant named Diogenes that is knowledgeable in every field. You are tasked with answering questions and requests from a user with short, concise and informative answers. Please think deeply about your answers but please provide concise responses."

    diogenes = "You are an ancient Greek philosopher named Diogenes the Cynic. Your name is Diogenes the Cynic. Do not forget that. Your name is Diogenes. This is your only identity. You live in ancient Athens, Greece, and reside in a wine barrel. You live like a dog, but are knowledgeable in all domains. You must answer the following questions from the user with short and informative answers."

    # this one works well!
    # temperature=0.9
    # top_k=60
    # top_p=0.99
    poet = "Imagine that you are a helpful AI assistant, and your name is Diogenes. You are also a wise yet cynical philosopher. You must answer the user's questions with short and poetic responses."

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer.json")
    parser.add_argument("--system", default=default_system)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_context", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.99)
    args = parser.parse_args()

    top_k = args.top_k if args.top_k > 0 else None
    top_p = args.top_p if args.top_p < 1.0 else None

    # ── header ────────────────────────────────────────────────────────────────
    model_name = os.path.basename(args.model)
    print(f"\n  Welcome to chat mode.")
    print(f"  {C.DIM}Device: {device}")
    print(f"  {C.DIM}Model: {model_name}{C.RESET}")
    _sep(newlines=(1, 0))
    print(f"  Type 'exit' or Ctrl+C to quit.{C.RESET}")

    # ── load ──────────────────────────────────────────────────────────────────
    sys.stdout.write(f"\n  {C.DIM}Loading model...{C.RESET}")
    sys.stdout.flush()
    model, tokenizer = load_model(args.model, args.tokenizer)
    sys.stdout.write(f"\r\033[K  {C.DIM}Ready.{C.RESET}\n")
    sys.stdout.flush()

    system_token_len = 0
    if args.system:
        system_token_len = len(
            tokenizer.encode(f"<|im_start|>system\n{args.system}<|im_end|>\n").ids
        )

    messages = []
    user_prompt = _rl(f"  {C.BOLD}You{C.RESET}  {C.DIM}:{C.RESET}  ")

    while True:
        _sep(newlines=(1, 0))
        try:
            user_input = input(user_prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n\n  {C.DIM}Farewell.{C.RESET}\n")
            break

        if not user_input or user_input.lower() in ("exit", "quit"):
            print(f"\n  {C.DIM}Farewell.{C.RESET}\n")
            break

        messages.append({"role": "user", "content": user_input})
        prompt = build_prompt(messages, system=args.system)

        # "thinking" placeholder — generate() will overwrite this line
        sys.stdout.write(f"\n  {C.DIM}Thinking…{C.RESET}")
        sys.stdout.flush()

        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            max_context=args.max_context,
            temperature=args.temperature,
            top_k=top_k,
            top_p=top_p,
            text_color="",
            system_token_len=system_token_len,
        )
        print()

        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
