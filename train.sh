#!/bin/bash
#SBATCH --job-name=train
#SBATCH --account=ai-gpu
#SBATCH --nodes=1
#SBATCH --partition=l40s-8-gm384-c192-m1536
#SBATCH --time=3:00:00
#SBATCH --output=train.out
#SBATCH --error=train.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --mail-user victor.li@emory.edu
#SBATCH --mail-type BEGIN
#SBATCH --mail-type END
#SBATCH --mail-type FAIL

conda init bash > /dev/null 2>&1
source ~/.bashrc
conda run -n cs python3.11 train.py