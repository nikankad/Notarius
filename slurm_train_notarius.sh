#!/bin/bash
#SBATCH --job-name=notarius-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2,gpumem:12G
#SBATCH --output=logs/slurm-%j.out

cd /home/nikan.kadkhodazadeh/Notarius || exit 1

source ~/.bashrc
source NotariusVenv/bin/activate

python --version
which python
python -c "import torch; print(torch.__version__)"
nvidia-smi

TORCHELASTIC_ERROR_FILE=/tmp/torch_elastic_error.json python -m torch.distributed.run --standalone --nproc_per_node=2 model/train_notarius.py \
  --epochs 50 \
  --R 3 \
  --C 192 \
  --expand 2 \
  --lr 0.005 \
  --warmup 2 \
  --batch-size 180 \
  --output-dir outputs/notarius \
  --log-aug-speccutout
