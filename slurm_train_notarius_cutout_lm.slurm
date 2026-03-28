#!/bin/bash
#SBATCH --job-name=notarius-cutout-lm
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:2,gpumem:40G
#SBATCH --output=logs/slurm-%j.out

cd "$HOME/Notarius" || exit 1

source ~/.bashrc
conda activate NotariusConda

python --version
which python
python -c "import torch; print(torch.__version__)"
nvidia-smi

torchrun --standalone --nproc_per_node=2 model/train_notarius_cutout_lm.py \
  --epochs 50 \
  --R 3 \
  --C 192 \
  --expand 2 \
  --lr 0.005 \
  --warmup 2 \
  --batch-size 180 \
  --output-dir outputs/notarius \
  --num-workers 8 \
  --arpa model/lm/6gram.arpa \
  --beam-width 100 \
  --lm-alpha 0.5 \
  --lm-beta 1.5
