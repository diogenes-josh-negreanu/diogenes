import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import torch
import signal
from functools import partial
from itertools import cycle
from datetime import datetime

from torch import nn, optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tokenizers import Tokenizer
from tqdm import tqdm
import wandb
import gc

from models.GPT import GPT
from data.SFTDataset import SFTDataset


# dynamically select device
if torch.cuda.is_available():
    device = torch.device("cuda")
    gc.collect()
    torch.cuda.empty_cache()
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


"""
Training and model configurations.
Can be changed prior to training.
"""
train_config = {
    'chunk_len': 1024,
    'bs': 8,
    'gradient_accumulation_steps': 4,
    'lr': 2e-5,
    'weight_decay': 0.01,
    'max_iters': 100_000,
    'warmup_iters': 100,
    'checkpoint_interval': 1000,
    'gpt_checkpoint': None,
    'sft_checkpoint': None,
}

model_config = {
    'emb_dim': 768,
    'num_layers': 12,
    'num_heads': 12
}


"""
interrupt_handler
    Saves a full-model checkpoint when training is interrupted via
    SIGINT or SIGTERM so that progress is not lost.

Args:
    iteration: int current step index
    loss: float or None most recent loss value
    model: torch.nn.Module language model
    tokenizer_path: string path to the saved tokenizer JSON
    scheduler: learning rate scheduler
    optimizer: torch optimizer
    train_config: dict training hyperparameters
    model_config: dict model hyperparameters
    project_name: string wandb project name
    run_name: string wandb run name
    sig: int signal number (required by signal handler contract)
    frame: frame object (required by signal handler contract)
"""
def interrupt_handler(
    iteration,
    loss,
    model,
    tokenizer_path,
    scheduler,
    optimizer,
    train_config,
    model_config,
    project_name,
    run_name,
    sig,
    frame
):
    torch.save({
        'iteration': iteration,
        'loss': loss,
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'tokenizer_path': tokenizer_path,
        'scheduler_state_dict': scheduler.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_config': train_config},
        f"./checkpoints/{project_name}/{run_name}/{run_name}_step{iteration}_int.pth"
    )


"""
train
    Full-model fine-tuning on the SFT dataset using cross-entropy loss
    with the pre-computed loss mask (non-assistant positions are -100
    and automatically ignored by CrossEntropyLoss).

    All model parameters are trained. Checkpoints save the full
    model_state_dict in the same format as pretraining checkpoints,
    so they can be loaded directly by chat.py.

Args:
    model: torch.nn.Module with pretrained weights already loaded
    dataloader: torch.utils.data.DataLoader yielding (inp, labels) pairs
    tokenizer_path: string path to the saved tokenizer JSON
    start_iter: int step to resume from (default 0)
    optimizer_state: optional dict from a prior checkpoint
    scheduler_state: optional dict from a prior checkpoint
"""
def train(model, dataloader, tokenizer_path, start_iter=0, optimizer_state=None, scheduler_state=None):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[train] training all {total_params:,} parameters")

    now = datetime.now()
    project_name = "sft"
    run_name = "sft" + now.strftime("%Y_%m_%d_%H_%M")
    wandb.login()
    wandb.init(project=project_name, name=run_name, config={**train_config, **model_config})
    os.makedirs(f"./checkpoints/{project_name}/{run_name}", exist_ok=True)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config['lr'],
        weight_decay=train_config['weight_decay']
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    warmup_iters   = train_config['warmup_iters']
    cooldown_iters = train_config['max_iters'] - warmup_iters
    linear  = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_iters)
    cosine  = CosineAnnealingLR(optimizer, T_max=cooldown_iters, eta_min=train_config['lr'] * 0.1)
    scheduler = SequentialLR(optimizer, schedulers=[linear, cosine], milestones=[warmup_iters])

    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    model.train()

    accum_steps = train_config.get('gradient_accumulation_steps', 1)
    data_iter   = cycle(dataloader)
    pbar = tqdm(total=train_config['max_iters'], initial=start_iter, desc="SFT Training", unit="step")

    for iteration in range(start_iter, train_config['max_iters']):
        signal.signal(signal.SIGINT, partial(interrupt_handler,
            iteration, None,
            model, tokenizer_path,
            scheduler, optimizer,
            train_config, model_config,
            project_name, run_name))
        signal.signal(signal.SIGTERM, partial(interrupt_handler,
            iteration, None,
            model, tokenizer_path,
            scheduler, optimizer,
            train_config, model_config,
            project_name, run_name))

        wandb.log({'learning_rate': scheduler.get_last_lr()[0]}, step=iteration)

        optimizer.zero_grad()
        total_loss = 0.0

        for _ in range(accum_steps):
            inp, labels = next(data_iter)
            inp, labels = inp.to(device), labels.to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                out  = model(inp)
                loss = criterion(out.permute(0, 2, 1), labels) / accum_steps
            loss.backward()
            total_loss += loss.item()

        wandb.log({"loss": total_loss}, step=iteration)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        pbar.update(1)

        if (iteration + 1) % train_config['checkpoint_interval'] == 0:
            torch.save({
                'iteration': iteration + 1,
                'loss': total_loss,
                'model_state_dict': model.state_dict(),
                'model_config': model_config,
                'tokenizer_path': tokenizer_path,
                'scheduler_state_dict': scheduler.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_config': train_config},
                f"./checkpoints/{project_name}/{run_name}/{run_name}_step{iteration + 1}.pth"
            )
            torch.cuda.empty_cache()

    wandb.finish()
    pbar.close()


"""
main
    Loads the pretrained checkpoint, builds the model, then enters the
    full-model SFT training loop. If sft_checkpoint is set in train_config,
    resumes from that checkpoint instead of the pretrained base.
"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer.json")
    parser.add_argument("--tokens", type=str, default="data/tokenized/sft/smoltalk.bin")
    parser.add_argument("--mask", type=str, default="data/tokenized/sft/smoltalk_mask.bin")
    args = parser.parse_args()

    tokenizer_path = args.tokenizer
    tokens_path = args.tokens
    mask_path = args.mask

    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()

    dataset = SFTDataset(tokens_path, mask_path, chunk_len=train_config['chunk_len'])
    dataloader = dataset.create_dataloader(bs=train_config['bs'])

    model = GPT(
        vocab_size=vocab_size,
        embed_dim=model_config['emb_dim'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        is_causal=True
    ).to(device)

    start_iter = 0
    optimizer_state = None
    scheduler_state = None

    sft_checkpoint = train_config.get('sft_checkpoint')
    if sft_checkpoint:
        # resume from finetuning training session
        chkpt = torch.load(sft_checkpoint, weights_only=False, map_location=device)
        model.load_state_dict(chkpt['model_state_dict'])
        optimizer_state = chkpt.get('optimizer_state_dict')
        scheduler_state = chkpt.get('scheduler_state_dict')
        start_iter = chkpt.get('iteration', 0)
        print(f"[main] resumed finetuning from step {start_iter}: {sft_checkpoint}")
    else:
        # start from the pretrained base
        chkpt = torch.load(
            train_config['gpt_checkpoint'],
            weights_only=False,
            map_location=device
        )
        model.load_state_dict(chkpt['model_state_dict'])
        print(f"[main] loaded pretrained checkpoint: {train_config['gpt_checkpoint']}")

    train(model, dataloader, tokenizer_path, start_iter, optimizer_state, scheduler_state)


if __name__ == '__main__':
    main()
