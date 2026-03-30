#!/bin/bash
#SBATCH --job-name=eval-lm
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:a100:1,gpumem:40G
#SBATCH --output=logs/slurm-%j.out

cd "$HOME/Notarius" || exit 1

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate NotariusConda

python --version
which python
nvidia-smi

export ROOT="$HOME/Notarius/datasets"

RESULTS_CSV="outputs/eval_results.csv"
ARPA="model/lm/6gram.arpa"
WEIGHTS="outputs/model_weights"

# # === Notarius ===
# python model/eval_lm.py --checkpoint "$WEIGHTS/Notarius/notarius-raw-best.pt"          --arpa "$ARPA" --beam-width 100 --lm-alpha 0.5 --lm-beta 1.5 --csv "$RESULTS_CSV"
# python model/eval_lm.py --checkpoint "$WEIGHTS/Notarius/notarius-perturb.pt"            --arpa "$ARPA" --beam-width 100 --lm-alpha 0.5 --lm-beta 1.5 --csv "$RESULTS_CSV"
# python model/eval_lm.py --checkpoint "$WEIGHTS/Notarius/notarius_cutout_best.pt"        --arpa "$ARPA" --beam-width 100 --lm-alpha 0.5 --lm-beta 1.5 --csv "$RESULTS_CSV"
# python model/eval_lm.py --checkpoint "$WEIGHTS/Notarius/notarius_cutout_speed_best.pt"  --arpa "$ARPA" --beam-width 100 --lm-alpha 0.5 --lm-beta 1.5 --csv "$RESULTS_CSV"

# === QuartzNet 10x5 ===
python model/eval_lm.py --checkpoint "$WEIGHTS/QN/QN10x5-raw_best.pt"           --arpa "$ARPA" --beam-width 100 --lm-alpha 0.5 --lm-beta 1.5 --csv "$RESULTS_CSV"
python model/eval_lm.py --checkpoint "$WEIGHTS/QN/QN10x5_cutout_best.pt"        --arpa "$ARPA" --beam-width 100 --lm-alpha 0.5 --lm-beta 1.5 --csv "$RESULTS_CSV"
python model/eval_lm.py --checkpoint "$WEIGHTS/QN/QN10x5_cutout_speed_best.pt"  --arpa "$ARPA" --beam-width 100 --lm-alpha 0.5 --lm-beta 1.5 --csv "$RESULTS_CSV"

# === QuartzNet 5x5 ===
python model/eval_lm.py --checkpoint "$WEIGHTS/QN/QN5x5-perturb_best.pt"        --arpa "$ARPA" --beam-width 100 --lm-alpha 0.5 --lm-beta 1.5 --csv "$RESULTS_CSV"
python model/eval_lm.py --checkpoint "$WEIGHTS/QN/QN5x5-cutout_best.pt"         --arpa "$ARPA" --beam-width 100 --lm-alpha 0.5 --lm-beta 1.5 --csv "$RESULTS_CSV"
python model/eval_lm.py --checkpoint "$WEIGHTS/QN/Qn5x5_cutout_speed_best.pt"   --arpa "$ARPA" --beam-width 100 --lm-alpha 0.5 --lm-beta 1.5 --csv "$RESULTS_CSV"
