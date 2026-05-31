#!/bin/bash -l
#SBATCH --job-name=gen-sqzero-sup-easy-10k
#SBATCH --partition=preempt
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=/hnvme/workspace/v123be13-WS/logs/%x-%j.out
#SBATCH --error=/hnvme/workspace/v123be13-WS/logs/%x-%j.err
#SBATCH --export=NONE

set -euo pipefail
unset SLURM_EXPORT_ENV

export WS=/hnvme/workspace/v123be13-WS
export REPO="$WS/point-mae-zero"
export OUT_ROOT="$WS/data/SQ_Zero"
export DATASET_NAME="sqzero_sup_easy_10k"

export N_CLOUDS=10000
export N_SQ_MAX=4
export N_POINTS_PER_CLOUD=4096
export SEED=12345

mkdir -p "$WS/logs"
mkdir -p "$OUT_ROOT"

cd "$REPO"

module load python
source /apps/python/3.12-conda/etc/profile.d/conda.sh
conda activate "$WS/conda_envs/pmaez"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

echo "=== Environment ==="
hostname
date
echo "WS=$WS"
echo "REPO=$REPO"
echo "OUT_ROOT=$OUT_ROOT"
echo "DATASET_NAME=$DATASET_NAME"
echo "N_CLOUDS=$N_CLOUDS"
echo "N_SQ_MAX=$N_SQ_MAX"
echo "N_POINTS_PER_CLOUD=$N_POINTS_PER_CLOUD"
echo "SEED=$SEED"
which python
python -c "import sys; print(sys.executable)"

echo "=== Check files ==="
ls -l procedural_data_gen/generate_sq_zero.py
ls -l procedural_data_gen/sample_SQs.py

echo "=== Script help / CLI check ==="
python procedural_data_gen/generate_sq_zero.py --help

echo "=== Verify new range CLI flags exist ==="
python procedural_data_gen/generate_sq_zero.py --help | grep -E "ax-range|eps1-range|tx-range"

echo "=== Clean previous output for this dataset ==="
rm -rf "$OUT_ROOT/$DATASET_NAME"

echo "=== Start SQ-Zero supervised-friendly generation ==="
date

srun python -u procedural_data_gen/generate_sq_zero.py \
  --out-dir "$OUT_ROOT" \
  --n-clouds "$N_CLOUDS" \
  --dataset-name "$DATASET_NAME" \
  --n-SQ-max "$N_SQ_MAX" \
  --n-points-per-cloud "$N_POINTS_PER_CLOUD" \
  --storage lmdb \
  --mode enriched \
  --sampling naive \
  --train-ratio 0.90 \
  --shuffle-split \
  --seed "$SEED" \
  --ax-range 0.15 1.0 \
  --ay-range 0.15 1.0 \
  --az-range 0.15 1.0 \
  --eps1-range 0.3 1.7 \
  --eps2-range 0.3 1.7 \
  --eulerx-range 0.0 6.283185307179586 \
  --eulery-range 0.0 6.283185307179586 \
  --eulerz-range 0.0 6.283185307179586 \
  --tx-range -0.8 0.8 \
  --ty-range -0.8 0.8 \
  --tz-range -0.8 0.8

echo "=== Finished SQ-Zero supervised-friendly generation ==="
date

echo "=== Output files ==="
find "$OUT_ROOT/$DATASET_NAME" -maxdepth 5 -type f -print | head -100 || true

echo "=== Disk usage ==="
du -sh "$OUT_ROOT/$DATASET_NAME" || true

echo "=== Inspect metadata ==="
cat "$OUT_ROOT/$DATASET_NAME"/shard_00000/metadata.json || true

echo "=== Done ==="
date
