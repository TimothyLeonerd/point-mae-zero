#!/bin/bash -l
#SBATCH --job-name=gen-sqzero-150k-pilu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=23:30:00
#SBATCH --output=%x-%j.out
#SBATCH --export=NONE

set -euo pipefail
unset SLURM_EXPORT_ENV

# --- go to repo ---
cd v123be13@helma6:/hnvme/workspace/v123be13-WS/point-mae-zero

conda activate pmaez

# helpful for CPU-bound runs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

# --- run ---
srun python procedural_data_gen/generate_sq_zero.py \
  --out-dir /hnvme/workspace/v123be13-WS/data/SQ_Zero/1500/pilu \
  --n-clouds 1500 \
  --dataset-name 1500_pilu \
  --sampling pilu
