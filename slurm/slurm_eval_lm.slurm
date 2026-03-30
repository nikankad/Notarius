#!/bin/bash
#SBATCH --job-name=eval-lm
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a100:1,gpumem:40G
#SBATCH --output=logs/slurm-%j.out

cd "$HOME/Notarius" || exit 1

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate NotariusConda

python --version
which python
nvidia-smi

RESULTS_CSV="outputs/eval_results.csv"

# === Evaluate Notarius (cutout) with LM ===
python model/eval_lm.py \
  --checkpoint outputs/notarius/notarius-cutout-242727/best.pt \
  --arpa model/lm/6gram.arpa \
  --beam-width 100 \
  --lm-alpha 0.5 \
  --lm-beta 1.5 \
  --csv "$RESULTS_CSV"

# === Evaluate QuartzNet (cutout) with LM ===
python model/eval_lm.py \
  --checkpoint outputs/quartznet/quartznet-cutout-242732/best.pt \
  --arpa model/lm/6gram.arpa \
  --beam-width 100 \
  --lm-alpha 0.5 \
  --lm-beta 1.5 \
  --csv "$RESULTS_CSV"
