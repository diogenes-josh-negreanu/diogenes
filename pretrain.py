import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

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

from models.LanguageTransformer import LanguageTransformer
from data.CorpusDataset import CorpusDataset


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
    'chunk_len': 2048,
    'bs': 8,
    'gradient_accumulation_steps': 4,
    'lr': 3e-4,
    'weight_decay': 0.1,
    'max_iters': 500_000,
    'warmup_iters': 1000,
    'checkpoint_interval': 5000,
    'checkpoint': None
}

model_config = {
    'emb_dim': 1024,
    'num_layers': 24,
    'num_heads': 16
}


"""
dry_run
    Runs the language model through random data to ensure proper
    dimensionality. Asserts correct output shape.

Args:
    model: torch.nn.Module language model
    bs: int batch size
    vocab_size: int size of vocabulary
    seq_len: int length of token sequence
"""
def dry_run(model, bs, vocab_size, seq_len):
    model.eval()
    with torch.no_grad():
        seq = torch.randint(0, vocab_size, (bs, seq_len)).to(device)
        out = model(seq)
        assert out.shape == (bs, seq_len, vocab_size)
    model.train()
    print("[dry_run] passed")


"""
interrupt_handler
    Saves a model checkpoint when training is interrupted via
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
        'tokenizer_path': tokenizer_path,
        'scheduler_state_dict': scheduler.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_config': train_config,
        'model_config': model_config},
        f"./checkpoints/{project_name}/{run_name}/{run_name}_step{iteration}_int.pth"
    )


"""
train
    Sets up wandb logging, creates an AdamW optimizer with linear
    warmup and cosine annealing schedule, and trains for max_iters
    gradient steps. Saves a checkpoint every checkpoint_interval steps.

Args:
    model: torch.nn.Module language model
    dataloader: torch.utils.data.DataLoader training data
    tokenizer_path: string path to the saved tokenizer JSON
"""
def train(model, dataloader, tokenizer_path, start_iter=0, optimizer_state=None, scheduler_state=None):
    # set up wandb and checkpoint path
    now = datetime.now()
    project_name = "diogenes-beta"
    run_name = "pretrain" + now.strftime("%Y_%m_%d_%H_%M")
    wandb.login()
    wandb.init(project=project_name, name=run_name, config={**train_config, **model_config})
    os.makedirs(f"./checkpoints/{project_name}/{run_name}", exist_ok=True)

    # optimizer and criterion
    optimizer = optim.AdamW(model.parameters(), lr=train_config['lr'], weight_decay=train_config['weight_decay'])
    criterion = nn.CrossEntropyLoss()

    # linear warmup for warmup_iters steps, then cosine decay to 10% of peak lr
    warmup_iters = train_config['warmup_iters']
    cooldown_iters = train_config['max_iters'] - warmup_iters

    linear = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_iters)
    cosine = CosineAnnealingLR(optimizer, T_max=cooldown_iters, eta_min=train_config['lr'] * 0.1)
    scheduler = SequentialLR(optimizer, schedulers=[linear, cosine], milestones=[warmup_iters])

    # restore optimizer and scheduler state when resuming from a checkpoint
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    model.train()

    accum_steps = train_config.get('gradient_accumulation_steps', 1)

    # cycle the dataloader so we can train for an arbitrary number of steps
    data_iter = cycle(dataloader)
    pbar = tqdm(total=train_config['max_iters'], initial=start_iter, desc="Training", unit="step")

    for iteration in range(start_iter, train_config['max_iters']):
        # update interrupt handler with current step so checkpoint filename is accurate
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

        # gradient accumulation: accumulate gradients over accum_steps micro-batches
        optimizer.zero_grad()
        total_loss = 0.0
        for _ in range(accum_steps):
            # batch shape: (B, chunk_len + 1)
            batch = next(data_iter).to(device)
            inp = batch[:, :-1]
            labels = batch[:, 1:]

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                out = model(inp, use_cache=False)
                loss = criterion(out.permute(0, 2, 1), labels) / accum_steps
            loss.backward()
            total_loss += loss.item()

        wandb.log({"loss": total_loss}, step=iteration)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        pbar.update(1)

        # save checkpoint every checkpoint_interval steps
        if (iteration + 1) % train_config['checkpoint_interval'] == 0:
            torch.save({
                'iteration': iteration + 1,
                'loss': total_loss,
                'model_state_dict': model.state_dict(),
                'tokenizer_path': tokenizer_path,
                'scheduler_state_dict': scheduler.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_config': train_config,
                'model_config': model_config},
                f"./checkpoints/{project_name}/{run_name}/{run_name}_step{iteration + 1}.pth"
            )
            torch.cuda.empty_cache()

    wandb.finish()
    pbar.close()


"""
main
    Loads the tokenizer and binary token dataset, builds the model,
    validates with a dry run, then enters the training loop.
"""
def main():
    # train on bookcorpus
    data_path = "data/pretrain/cosmopedia-v2.bin"
    tokenizer_path = "data/tokenizers/tokenizer.json"

    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()

    dataset = CorpusDataset(data_path, chunk_len=train_config['chunk_len'])
    dataloader = dataset.create_dataloader(bs=train_config['bs'])

    model = LanguageTransformer(
        vocab_size=vocab_size,
        embed_dim=model_config['emb_dim'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        is_causal=True
    ).to(device)

    start_iter = 0
    optimizer_state = None
    scheduler_state = None
    if train_config['checkpoint']:
        chkpt = torch.load(train_config['checkpoint'], weights_only=False, map_location=device)
        model.load_state_dict(chkpt['model_state_dict'])
        # optimizer_state = chkpt.get('optimizer_state_dict')
        # scheduler_state = chkpt.get('scheduler_state_dict')
        # start_iter = chkpt.get('iteration', 0)
        # print(f"[resume] loaded checkpoint from iteration {start_iter}")

    dry_run(model, train_config['bs'], vocab_size, train_config['chunk_len'])
    train(model, dataloader, tokenizer_path, start_iter=start_iter,
          optimizer_state=optimizer_state, scheduler_state=scheduler_state)


if __name__ == '__main__':
    main()
