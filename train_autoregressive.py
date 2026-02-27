import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import math
import pandas as pd
import random
import wandb

import sys
import signal
from functools import partial

from torch import nn, optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, ConcatDataset

from datasets import load_dataset
from transformers import AutoTokenizer
from staticvectors import StaticVectors
from datetime import datetime
from tqdm import tqdm

from models.LanguageTransformer import LanguageTransformer
from data.AutoregressiveLanguage import AutoregressiveLanguageDataset

# dynamically select device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


"""
Training  model and configurations.
Can be changed prior to training.
"""
train_config = {
    'max_examples': 500000,
    'max_len': 1000,
    'bs': 32,
    'lr': 0.0001,
    'weight_decay': 0.000001,
    'max_epochs': 10
}

model_config = {
    'emb_dim': 256,
    'num_layers': 8,
    'num_heads': 8
}


"""
dry_run
    Runs language model through random to ensure proper
    dimensionality. Asserts correct shape.

    Args:
        model: torch.nn.Module language model
        bs: int batch size
        vocab_len: int size of vocab
        seq_len: int length of token sequence
"""
def dry_run(model, bs, vocab_len, seq_len):
    seq = torch.randint(0, vocab_len, (bs, seq_len)).to(device)
    out = model(seq)
    assert out.shape == (bs, seq_len, vocab_len)
    print("[dry_run] passed")


"""
interrupt_handler
    Save model checkpoint in case of terminal interrupt.
    Special checkpoint file tag not to override current
    epoch checkpoint.

Args:
    A ton haha...
"""
def interrupt_handler(
    epoch,
    loss,
    model,
    vocab,
    scheduler,
    optimizer,
    train_config,
    model_config,
    project_name,
    run_name,
    sig,
    frame
):
    # save model each epoch
        torch.save({
            'epoch': epoch,
            'loss': loss,
            'model_state_dict': model.state_dict(),
            'vocab': vocab,
            'scheduler_state_dict': scheduler.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_config': train_config,
            'model_config': model_config},
            f"./checkpoints/{project_name}/{run_name}/{run_name}_epoch{epoch}_int.pth"
        )


"""
train
    Sets up wandb logging, creates AdamW optimizer,
    custom linear warmup and cosine annealing scheduler,
    trains for designated number of epochs, saving model
    checkpoints each epoch.

    Args:
        model: torch.nn.Module language model
        data_loader: torch.DataLoader training data
"""
def train(model, dataloader, vocab):
    # set up wandb and checkpoint path
    now = datetime.now()
    project_name = "autoregressive-language-model"
    run_name = "alm-" + now.strftime("%Y_%m_%d_%H_%m")
    wandb.login()
    wandb.init(project=project_name, name=run_name, config=train_config)
    os.makedirs(f"./checkpoints/{project_name}/{run_name}", exist_ok=True)

    # optimizer and criterion
    optimizer = optim.AdamW(model.parameters(), lr=train_config['lr'], weight_decay=train_config['weight_decay'])
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # construct linear warmup and cosine annealing cooldown
    warmup_epochs = int(train_config['max_epochs'] / 10)
    cooldown_epochs = train_config['max_epochs'] - warmup_epochs
    epoch_len = len(dataloader)

    linear = LinearLR(optimizer, start_factor=0.25, end_factor=1.0, total_iters=warmup_epochs*epoch_len)
    cosine = CosineAnnealingLR(optimizer, T_max=cooldown_epochs*epoch_len, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[linear, cosine], milestones=[warmup_epochs*epoch_len])

    model.train()

    # main training loop
    pbar = tqdm(total=(train_config['max_epochs'])*epoch_len, desc="Training Iterations", unit="batch")
    iteration = 0
    for epoch in range(train_config['max_epochs']):
        # signal catching to save model on interrupt
        signal.signal(signal.SIGINT, partial(interrupt_handler,
            epoch, None,
            model, vocab,
            scheduler, optimizer,
            train_config, model_config,
            project_name, run_name))
        signal.signal(signal.SIGTERM, partial(interrupt_handler,
            epoch, None,
            model, vocab,
            scheduler, optimizer,
            train_config, model_config,
            project_name, run_name))

        # minibatch gradient descent
        epoch_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            wandb.log({'learning-rate': scheduler.get_last_lr()[0]}, step=iteration)

            batch = batch.to(device)
            out = model(batch)[:,:-1,:]
            labels = batch[:,1:]

            # compute loss
            loss = criterion(out.permute(0, 2, 1), labels)
            epoch_loss += loss
            wandb.log({"loss": loss.item()}, step=iteration)

            # optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            pbar.update(1)
            iteration += 1
            scheduler.step()

        # save model each epoch
        torch.save({
            'epoch': epoch,
            'loss': epoch_loss / epoch_len,
            'model_state_dict': model.state_dict(),
            'vocab': vocab,
            'scheduler_state_dict': scheduler.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_config': train_config,
            'model_config': model_config},
            f"./checkpoints/{project_name}/{run_name}/{run_name}_epoch{epoch}_end.pth"
        )

    wandb.finish()
    pbar.close()


"""
main
    Builds a model, checks through a dry run, runs
    through training cycle.
"""
def main():
    # create dataset
    dataset = AutoregressiveLanguageDataset(
        dataset_name="roneneldan/TinyStories",
        max_examples=train_config['max_examples'],
        max_len=train_config['max_len'],
        bs=train_config['bs']
    )

    # create language model
    model = LanguageTransformer(
        vocab_size=len(dataset.vocab),
        embed_dim=model_config['emb_dim'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        is_causal=True
    ).to(device)
    dry_run(model, train_config['bs'], len(dataset.vocab), 100)

    # enter training cycle
    train(model, dataset.create_dataloader(), dataset.vocab)


if __name__ == '__main__':
    main()