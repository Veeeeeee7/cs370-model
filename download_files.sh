#!/bin/bash
#SBATCH --job-name=downloader
#SBATCH --account=general
#SBATCH --nodes=1
#SBATCH --partition=c64-m512
#SBATCH --time=3:00:00
#SBATCH --output=download_files.out
#SBATCH --error=download_files.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-user victor.li@emory.edu
#SBATCH --mail-type BEGIN
#SBATCH --mail-type END
#SBATCH --mail-type FAIL

conda init bash > /dev/null 2>&1
source ~/.bashrc
conda run -n cs python3.11 download_files.py