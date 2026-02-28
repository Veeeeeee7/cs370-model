#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --account=ai-gpu
#SBATCH --nodes=1
#SBATCH --partition=l40s-8-gm384-c192-m1536
#SBATCH --time=3:0:00
#SBATCH --output=inference.out
#SBATCH --error=inference.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --mail-user victor.li@emory.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

conda init bash > /dev/null 2>&1
source ~/.bashrc

# ---- EDIT THESE ----
IMAGE="/scratch/vmli3/cs370/data/part0/sub067/CA-25_03_09_2015_CA-25_0000237.jpg"
CKPT="/scratch/vmli3/cs370/model-checkpoints/best.pt"
# --------------------

conda run -n cs python3.11 inference.py \
  --image "$IMAGE" \
  --checkpoint "$CKPT" \
  --clamp0 --round \
  --copy-to "."

# RUN THIS IN TERMINAL LOCALLY TO TEST (change the image path if needed)
# python inference.py --image "nacti_subset/part0/sub042/CA-10_03_10_2015_CA-10_0000103.jpg" --checkpoint "best.pt" --clamp0 --round --copy-to "."