#!/bin/bash -l
#SBATCH --job-name=gen-sqzero-150k-pilu
#SBATCH --partition=preempt_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=23:30:00
#SBATCH --output=%x-%j.out
#SBATCH --export=NONE

set -euo pipefail
unset SLURM_EXPORT_ENV

# --- go to repo ---
cd /home/atuin/v123be/v123be13/Master_Arbeit/point-mae-zero

# --- activate conda env (robust across setups) ---
# Try common conda locations; adjust if yours lives elsewhere
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  # conda is already on PATH; try to hook shell functions
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  echo "ERROR: conda not found. Edit this script to source your conda.sh." >&2
  exit 1
fi

conda activate pmaez

# helpful for CPU-bound runs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

# --- run ---
srun python procedural_data_gen/generate_sq_zero.py \
  --out-dir /home/atuin/v123be/v123be13/data/SQ_Zero/150k/150k_pilu \
  --n-clouds 150000 \
  --dataset-name 150k_pilu \
  --sampling pilu
