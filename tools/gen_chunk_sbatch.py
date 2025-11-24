#!/usr/bin/env python
import argparse
import math
import os
from datetime import datetime

import yaml


def hhmmss_from_minutes(minutes: float) -> str:
    total_seconds = max(60, int(round(minutes * 60)))  # at least 1 min
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    parser = argparse.ArgumentParser(
        description="Generate chunked Slurm sbatch scripts for Point-MAE-Zero pretraining."
    )
    parser.add_argument("--config", required=True, help="YAML config, e.g. cfgs/pretrain_*.yaml")
    parser.add_argument("--exp-dir", default="experiments", help="Experiment root directory (relative to repo root).")
    parser.add_argument("--exp-name", help="Experiment name (folder under exp-dir). If omitted, generated from config+timestamp.")
    parser.add_argument("--epochs-per-chunk", type=int, required=True, help="Approximate epochs per chunk (for time estimation).")
    parser.add_argument("--est-mins-per-epoch", type=float, required=True, help="Estimated minutes per epoch.")
    parser.add_argument("--safety-factor", type=float, default=1.3, help="Safety multiplier for wall time.")
    parser.add_argument("--partition", default="a100", help="Slurm partition (e.g. a100, a40).")
    parser.add_argument("--gres", help="Slurm --gres string, e.g. gpu:a100:1 (default gpu:<partition>:1).")
    parser.add_argument("--cpus-per-task", type=int, default=12, help="Slurm --cpus-per-task.")
    args = parser.parse_args()

    # Detect repo root (this script is assumed to live in tools/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)

    config_abs = os.path.join(repo_root, args.config)
    if not os.path.isfile(config_abs):
        raise FileNotFoundError(f"Config not found: {config_abs}")

    # Read YAML to get max_epoch
    with open(config_abs, "r") as f:
        cfg = yaml.safe_load(f)
    if "max_epoch" not in cfg:
        raise KeyError(f"'max_epoch' not found in config {args.config}")
    max_epoch = int(cfg["max_epoch"])

    # Determine experiment name
    if args.exp_name:
        exp_name = args.exp_name
    else:
        stem = os.path.splitext(os.path.basename(args.config))[0]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{stem}_{ts}"

    exp_dir_rel = args.exp_dir
    exp_dir_abs = os.path.join(repo_root, exp_dir_rel)

    # Where to write sbatch files
    scripts_root = os.path.join(repo_root, "scripts")
    scripts_dir = os.path.join(scripts_root, exp_name)
    os.makedirs(scripts_dir, exist_ok=True)

    # Check if this experiment already has a ckpt-last.pth (resume-from-existing case)
    ckpt_last = os.path.join(exp_dir_abs, exp_name, "ckpt-last.pth")
    first_uses_resume = os.path.exists(ckpt_last)

    total_epochs = max_epoch
    epochs_per_chunk = args.epochs_per_chunk
    num_chunks = math.ceil(total_epochs / epochs_per_chunk)

    gres = args.gres or f"gpu:{args.partition}:1"
    config_rel_from_repo = os.path.relpath(config_abs, repo_root)

    print(f"[INFO] repo_root        = {repo_root}")
    print(f"[INFO] config           = {config_rel_from_repo} (max_epoch={max_epoch})")
    print(f"[INFO] exp_dir (rel)    = {exp_dir_rel}")
    print(f"[INFO] exp_name         = {exp_name}")
    print(f"[INFO] scripts dir      = {scripts_dir}")
    print(f"[INFO] num_chunks       = {num_chunks}")
    print(f"[INFO] first uses resume? {'YES' if first_uses_resume else 'NO'}")

    for i in range(1, num_chunks + 1):
        # for time estimate, assume linear splitting; doesn't need to be exact
        start_epoch_est = (i - 1) * epochs_per_chunk
        remaining = max(0, total_epochs - start_epoch_est)
        epochs_this_chunk = min(epochs_per_chunk, remaining) if remaining > 0 else epochs_per_chunk

        minutes = epochs_this_chunk * args.est_mins_per_epoch * args.safety_factor
        time_str = hhmmss_from_minutes(minutes)

        chunk_name = f"c{i}"
        sbatch_path = os.path.join(scripts_dir, f"{chunk_name}.sbatch")

        use_resume = first_uses_resume or i > 1

        job_name = f"{exp_name}_{chunk_name}"

        # Note: REPO is baked as an absolute path; this matches your current scripts,
        # and ensures the right directory is used even if you submit from elsewhere.
        sbatch_lines = [
            "#!/bin/bash -l",
            "#SBATCH --export=NONE",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH -p {args.partition}",
            f"#SBATCH --gres={gres}",
            f"#SBATCH --time={time_str}",
            f"#SBATCH --cpus-per-task={args.cpus_per_task}",
            "#SBATCH --output=%x-%j.out",
            "",
            "unset SLURM_EXPORT_ENV",
            "set -euo pipefail",
            "",
            "module load python",
            "module load cuda/11.8.0",
            "module load gcc/11",
            "source ~/.bashrc",
            "conda activate pmaez",
            "",
            "export OMP_NUM_THREADS=1",
            "export MKL_NUM_THREADS=1",
            "export PYTHONUNBUFFERED=1",
            "",
            "export TORCH_LIB_DIR=\"$(python - <<'PY'",
            "import os, torch",
            "print(os.path.join(os.path.dirname(torch.__file__), 'lib'))",
            "PY",
            ")\"",
            "export LD_LIBRARY_PATH=\"$TORCH_LIB_DIR:${LD_LIBRARY_PATH}\"",
            "",
            f"REPO=\"{repo_root}\"",
            "cd \"$REPO\"",
            "",
            f"EXP_DIR=\"$REPO/{exp_dir_rel}\"",
            f"EXP_NAME=\"{exp_name}\"",
            "",
            "mkdir -p exp",
            f"nvidia-smi dmon -s pucm -d 5 -f \"exp/{job_name}_${{SLURM_JOB_ID}}.log\" >/dev/null 2>&1 &",
            "DMON_PID=$!",
            "",
            f"echo \"[INFO] starting {job_name} at: $(date)\"",
            "START_TS=$(date +%s)",
            "",
            "stdbuf -oL -eL python -u main.py \\",
            f"  --config {config_rel_from_repo} \\",
            f"  --exp_dir \"$EXP_DIR\" \\",
            f"  --exp_name \"$EXP_NAME\" \\",
            f"  --job-max-epochs {epochs_per_chunk}" + (" \\" if use_resume else ""),
        ]

        if use_resume:
            sbatch_lines.append("  --resume")

        sbatch_lines.extend(
            [
                "",
                "STATUS=$?",
                "END_TS=$(date +%s)",
                "echo \"[INFO] finished with status=$STATUS at: $(date)\"",
                "echo \"[INFO] wall time: $((END_TS-START_TS)) seconds\"",
                "",
                "kill $DMON_PID 2>/dev/null || true",
                "exit $STATUS",
                "",
            ]
        )

        with open(sbatch_path, "w") as f:
            f.write("\n".join(sbatch_lines))

        print(f"[INFO] wrote {sbatch_path} (time={time_str}, use_resume={use_resume})")


if __name__ == "__main__":
    main()
