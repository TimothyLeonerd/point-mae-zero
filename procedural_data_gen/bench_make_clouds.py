# bench_make_clouds.py
import numpy as np
from collections import defaultdict
from sample_SQs import Prof, make_cloud_profiled

def merge_prof(dst, src):
    """Add section totals/call-counts from src into dst."""
    for k, v in src.t.items():
        dst.t[k] += v
    for k, v in src.n.items():
        dst.n[k] += v

def bench_make_clouds(
    trials=50,
    n_SQ_max=9,
    n_points=8192,
    alpha=2.0, growth=1.3, max_rounds=6,
    seed=42
):
    rng = np.random.default_rng(seed)

    overall = Prof()                 # aggregate over ALL runs
    per_n_sqs = defaultdict(Prof)    # aggregate per #SQs

    skipped = 0
    for _ in range(trials):
        try:
            out_pts, sq_pars, prof = make_cloud_profiled(
                n_SQ_max, n_points, rng=rng, alpha=alpha, growth=growth, max_rounds=max_rounds
            )
        except ValueError:
            skipped += 1
            continue

        # merge totals
        merge_prof(overall, prof)
        merge_prof(per_n_sqs[len(sq_pars)], prof)

    print(f"\n=== Overall (trials={trials}, skipped={skipped}) ===")
    overall.report()

    print("\n=== By number of SQs ===")
    for n_sqs in sorted(per_n_sqs):
        print(f"\n#SQs = {n_sqs}")
        per_n_sqs[n_sqs].report()

if __name__ == "__main__":
    bench_make_clouds(trials=50, n_SQ_max=9, n_points=8192)
