#!/bin/bash

#SBATCH --partition=dgxh

#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL

source /nfs/stak/users/negreanj/hpc-share/miniforge/etc/profile.d/conda.sh
conda init
conda activate torch_env

# python scripts/train_tokenizer.py

# data preparation scripts
# python scripts/prepare_pretrain_data.py
# python scripts/prepare_sft_data.py

# training scripts
python pretrain.py
# python train_sft.py