#!/bin/bash
#SBATCH --job-name=qn-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:4,gpumem:40G
#SBATCH --output=logs/slurm-%j.out

cd "$HOME/Notarius" || exit 1

source ~/.bashrc

# activate the environment that has torch installed
# example:
# conda activate notarius
# or:
source Notariusvenv/bin/activate

python --version
which python
python -c "import torch; print(torch.__version__)"
nvidia-smi

torchrun --standalone --nproc_per_node=4 model/train_qn_cutout.py \
  --epochs 50 \
  --B 10 \
  --R 5 \
  --lr 0.005 \
  --warmup 2 \
  --batch-size 180 \
  --output-dir outputs/quartznet \
  --num-workers 8\
  --log-aug-speccutout