#!/usr/bin/env python
import argparse
import os
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(
        description="Launch a dependency chain of chunked sbatch scripts."
    )
    parser.add_argument(
        "--exp-name",
        required=True,
        help="Experiment name whose scripts live under scripts/<exp-name>/c*.sbatch",
    )
    parser.add_argument(
        "--scripts-dir",
        default="scripts",
        help="Base scripts directory (default: scripts).",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)

    exp_scripts_dir = os.path.join(repo_root, args.scripts_dir, args.exp_name)
    if not os.path.isdir(exp_scripts_dir):
        raise FileNotFoundError(f"Scripts directory not found: {exp_scripts_dir}")

    # Find c*.sbatch files and sort by numeric index
    pattern = re.compile(r"^c(\d+)\.sbatch$")
    chunks = []
    for fname in os.listdir(exp_scripts_dir):
        m = pattern.match(fname)
        if m:
            idx = int(m.group(1))
            chunks.append((idx, fname))

    if not chunks:
        raise RuntimeError(f"No c*.sbatch files found in {exp_scripts_dir}")

    chunks.sort(key=lambda x: x[0])

    print(f"[INFO] Found {len(chunks)} chunk scripts in {exp_scripts_dir}:")
    for idx, fname in chunks:
        print(f"  - {fname} (chunk {idx})")

    prev_jobid = None
    for idx, fname in chunks:
        sbatch_path = os.path.join(exp_scripts_dir, fname)
        cmd = ["sbatch"]
        if prev_jobid is not None:
            cmd.append(f"--dependency=afterok:{prev_jobid}")
        cmd.append(sbatch_path)

        print(f"[INFO] Submitting {fname} with command: {' '.join(cmd)}")
        out = subprocess.check_output(cmd, text=True)
        out = out.strip()
        # Usually: "Submitted batch job 3129393"
        jobid = out.split()[-1]
        print(f"[INFO] {fname} -> job {jobid}")
        prev_jobid = jobid

    print("[INFO] All chunks submitted.")


if __name__ == "__main__":
    main()
