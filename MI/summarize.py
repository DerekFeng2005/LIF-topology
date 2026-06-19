"""
Extract last-epoch IZX/IZY from MI experiment logs, then per-seed L2
normalize within each topology group (4-LIF / 5-LIF), per LIF2HH paper
equation (26). Output per-group CSV (no comments, valid CSV).
"""
import os
import re
import math
from collections import defaultdict

RECORD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "record")
GROUPS = ["4-LIF", "5-LIF"]

TEST_RE = re.compile(r"e:(\d+)\s+IZY:([-\d.]+)\s+IZX:([-\d.]+)")
LOG_NAME_RE = re.compile(r"^(?P<model>.+?)_seed(?P<seed>\d+)\.log$")


def parse_last_epoch(path):
    last = None
    with open(path, "r") as f:
        for line in f:
            m = TEST_RE.search(line)
            if m:
                last = (int(m.group(1)), float(m.group(2)), float(m.group(3)))
    if last is None:
        raise ValueError(f"No [TEST RESULT] line in {path}")
    return last


def collect_group(group):
    group_dir = os.path.join(RECORD_DIR, group)
    by_seed = defaultdict(dict)
    for fname in sorted(os.listdir(group_dir)):
        m = LOG_NAME_RE.match(fname)
        if not m:
            continue
        model = m.group("model")
        seed = int(m.group("seed"))
        epoch, izy, izx = parse_last_epoch(os.path.join(group_dir, fname))
        by_seed[seed][model] = (epoch, izy, izx)
    return by_seed


def l2_normalize_per_seed(by_seed, metric_idx):
    normed = {}
    for seed, models in by_seed.items():
        vals = [v[metric_idx] for v in models.values()]
        denom = math.sqrt(sum(v * v for v in vals))
        normed[seed] = {
            model: (v[metric_idx] / denom if denom > 0 else 0.0)
            for model, v in models.items()
        }
    return normed


def mean_std(vals):
    n = len(vals)
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / n if n > 1 else 0.0
    return m, math.sqrt(var)


def write_group_csv(group, by_seed, out_path):
    izy_norm = l2_normalize_per_seed(by_seed, 1)
    izx_norm = l2_normalize_per_seed(by_seed, 2)

    with open(out_path, "w", newline="") as f:
        f.write("group,model,seed,epoch,"
                "IZY_raw_mean,IZX_raw_mean,"
                "IZY_norm_mean,IZX_norm_mean,"
                "IZY_norm_std,IZX_norm_std\n")

        # Per-seed rows (std columns empty)
        for seed in sorted(by_seed):
            for model in sorted(by_seed[seed]):
                epoch, izy, izx = by_seed[seed][model]
                f.write(
                    f"{group},{model},{seed},{epoch},"
                    f"{izy:.4f},{izx:.4f},"
                    f"{izy_norm[seed][model]:.6f},{izx_norm[seed][model]:.6f},"
                    f",\n"
                )

        # Mean rows: per-model, cross-seed average
        # and std rows: per-model, cross-seed std
        all_models = sorted({m for models in by_seed.values() for m in models})
        for model in all_models:
            seeds_with_model = sorted(s for s, models in by_seed.items() if model in models)
            epoch_ref = by_seed[seeds_with_model[0]][model][0]

            izy_raw_vals = [by_seed[s][model][1] for s in seeds_with_model]
            izx_raw_vals = [by_seed[s][model][2] for s in seeds_with_model]
            izy_n_vals = [izy_norm[s][model] for s in seeds_with_model]
            izx_n_vals = [izx_norm[s][model] for s in seeds_with_model]

            izy_raw_mean, _ = mean_std(izy_raw_vals)
            izx_raw_mean, _ = mean_std(izx_raw_vals)
            izy_n_mean, izy_n_std = mean_std(izy_n_vals)
            izx_n_mean, izx_n_std = mean_std(izx_n_vals)

            f.write(
                f"{group},{model},mean,{epoch_ref},"
                f"{izy_raw_mean:.4f},{izx_raw_mean:.4f},"
                f"{izy_n_mean:.6f},{izx_n_mean:.6f},"
                f"{izy_n_std:.6f},{izx_n_std:.6f}\n"
            )


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    for group in GROUPS:
        by_seed = collect_group(group)
        if not by_seed:
            print(f"[warn] no logs found in {group}")
            continue
        out_path = os.path.join(base, f"summary_{group}.csv")
        write_group_csv(group, by_seed, out_path)
        n_seeds = len(by_seed)
        n_models = len({m for models in by_seed.values() for m in models})
        print(f"[ok] {group}: {n_seeds} seeds, {n_models} models -> {out_path}")


if __name__ == "__main__":
    main()
