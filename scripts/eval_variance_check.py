#!/usr/bin/env python
"""Seed-to-seed variance across ctx_size (Sprint 01 / 1.4 follow-up).

Same sweep as eval_ctx_sweep.py -- ctx_size on the x-axis -- but repeats each
ctx_size --n-seeds times, changing only shuffle_seed between repeats
(shuffle_seed picks *which* row subset gets evaluated -- see the parameter
table in experiment.ipynb, Sprint 01 / 1.2). The spread across those repeats
at a given ctx_size is exactly the sampling noise in the metric at that point
-- --items-per-task is a single fixed value here (not swept) since it's the
thing meant to be "large enough" to keep that spread small.

Local smoke test (tiny + eager, to catch bugs fast without waiting on Slurm):
    pixi run python scripts/eval_variance_check.py \\
        --ctx-sizes 128 --items-per-task 1 --n-seeds 2 --disable-compile \\
        --out-dir /tmp/variance_test

Cluster run (mig-preempt, via scripts/eval_variance_check.sh -- do NOT pass
--disable-compile there, same reasoning as eval_ctx_sweep.py):
    pixi run python scripts/eval_variance_check.py \\
        --ctx-sizes 128 192 256 384 512 --items-per-task 16 --n-seeds 8

LOGGING: same convention as eval_ctx_sweep.py -- every log() line is
timestamped + elapsed-since-start and flush=True'd immediately.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # see eval_ctx_sweep.py -- same
# macOS OpenMP double-init fix, harmless no-op elsewhere.

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # see
# eval_ctx_sweep.py -- same fragmentation fix, harmless when it isn't needed.

import gc

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless -- this runs under Slurm / no display, always save to a file
import matplotlib.pyplot as plt
import torch
import torch._dynamo  # module level, not inside main() -- see eval_ctx_sweep.py's FIX note

from rt.checkpoints import load_rt_model
from rt.recipes import get_tasks
from rt.eval_utils import build_evaluator, metric_for

_T0 = time.monotonic()


def log(msg: str) -> None:
    elapsed = time.monotonic() - _T0
    print(f"[{datetime.now().strftime('%H:%M:%S')} +{elapsed:7.1f}s] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="stanford-star/rt-j/regression")
    ap.add_argument("--pre-dir", default="stanford-star/relbench-preprocessed")
    ap.add_argument("--task", default="rel-f1/driver-position",
                    help="'db/task-table' selector, as in scripts/eval.py --tasks")
    ap.add_argument("--ctx-sizes", type=int, nargs="+", default=[128, 192, 256, 384, 512])
    ap.add_argument("--items-per-task", type=int, default=16,
                    help="fixed rows sampled per run -- the thing meant to be large enough "
                         "to keep the across-seed spread small")
    ap.add_argument("--n-seeds", type=int, default=8,
                    help="repeats per ctx_size, each with a different shuffle_seed")
    ap.add_argument("--bfs-width", type=int, default=32)
    ap.add_argument("--local-ctx-size-cap", type=int, default=256,
                    help="local_ctx_size = min(this, ctx_size) for each point")
    ap.add_argument("--reg-metric", default="mae", choices=["mae", "r2"])
    ap.add_argument("--out-dir", default="cluster_run/variation_check")
    ap.add_argument("--disable-compile", action="store_true",
                    help="fall back to eager execution -- only pass this where the compiler "
                         "is broken (e.g. this Mac); leave it off on the cluster")
    args = ap.parse_args()
    log(f"starting: ctx_sizes={args.ctx_sizes} items_per_task={args.items_per_task} "
        f"n_seeds={args.n_seeds} task={args.task} disable_compile={args.disable_compile}")

    if args.disable_compile:
        torch._dynamo.config.disable = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"loading checkpoint on {device} ...")
    net, config = load_rt_model(args.checkpoint, device=device, compile=False)
    net = net.to(torch.bfloat16)
    log(f"loaded {config.get('name', args.checkpoint)} (task_type={config.get('task_type')})")

    all_tasks = get_tasks("relbench_eval_test", args.pre_dir)
    tasks = [t for t in all_tasks if f"{t.db_name}/{t.table_name}" == args.task]
    if not tasks:
        raise SystemExit(f"{args.task} not found in relbench_eval_test tasks for {args.pre_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # results.json is written after every single run (not just after each ctx_size), same
    # reasoning as eval_ctx_sweep.py -- a crash partway through never loses what's already
    # computed.
    results = []
    n_runs = len(args.ctx_sizes) * args.n_seeds
    run_i = 0
    for ctx_size in args.ctx_sizes:
        local_ctx_size = min(args.local_ctx_size_cap, ctx_size)
        for seed in range(args.n_seeds):
            run_i += 1
            log(f"[{run_i}/{n_runs}] ctx_size={ctx_size} shuffle_seed={seed} "
                f"-- building evaluator ...")
            # num_workers=0: build_evaluator() defaults to 2 DataLoader worker *subprocesses*
            # per Evaluator -- fine for eval_ctx_sweep.py's one-Evaluator-per-process-run, but
            # this loop builds up to 40 Evaluators in one process (5 ctx_sizes x 8 seeds) and
            # Evaluator exposes no close()/shutdown(), so cleanup relies on Python's GC timing.
            # A real run confirmed this: 17 runs completed fine, then a Slurm oom_kill (host
            # RAM, not GPU) on run 18 -- worker processes from earlier Evaluators piling up
            # faster than they were reclaimed. items_per_task is small here (16), so the
            # parallelism num_workers=2 buys is negligible -- not worth the leak risk.
            ev = build_evaluator(tasks, args.pre_dir, embedding_model=config["embedding_model"],
                                 d_text=config["d_text"], device=device, ctx_size=ctx_size,
                                 local_ctx_size=local_ctx_size, bfs_width=args.bfs_width,
                                 items_per_task=args.items_per_task, shuffle_seed=seed,
                                 num_workers=0)
            for task, _ctx, labels, preds_by_prefix, _nl in ev.evaluate_raw(
                [(net, "")], [ctx_size]
            ):
                metric_name, metric_value = metric_for(task.task_type, labels, preds_by_prefix[""],
                                                        reg_metric=args.reg_metric)
                n = len(labels)
            results.append({"ctx_size": ctx_size, "shuffle_seed": seed,
                            "metric": metric_name, "value": metric_value, "n": n})
            log(f"[{run_i}/{n_runs}] ctx_size={ctx_size} shuffle_seed={seed}: "
                f"{metric_name}={metric_value:.4f} (n={n}) -- done")
            (out_dir / "results.json").write_text(json.dumps(results, indent=2))

            # Cheap extra insurance on top of num_workers=0: force this Evaluator's memory
            # back before building the next one, rather than trust GC timing across 40 builds.
            del ev
            gc.collect()

    log(f"wrote {out_dir / 'results.json'}")

    # Plot: for each ctx_size, the individual per-seed values (dots) plus mean +- 1 std
    # (errorbar) -- same idea as the ctx_size sweep plot, but now showing the sampling
    # spread at each point instead of a single value.
    metric_name = results[0]["metric"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    means, stds = [], []
    for ctx_size in args.ctx_sizes:
        vals = [r["value"] for r in results if r["ctx_size"] == ctx_size]
        means.append(np.mean(vals))
        stds.append(np.std(vals))
        ax.scatter([ctx_size] * len(vals), vals, color="#a0aec0", zorder=2, s=25)
    ax.errorbar(args.ctx_sizes, means, yerr=stds, marker="o", capsize=4,
               color="#2b6cb0", zorder=3, label=f"mean ± 1 std across {args.n_seeds} seeds")
    ax.set_xscale("log", base=2)
    ax.set_xticks(args.ctx_sizes)
    ax.set_xticklabels(args.ctx_sizes)
    ax.set_xlabel("ctx_size")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{args.task}: {metric_name} spread across {args.n_seeds} shuffle_seeds\n"
                 f"at each ctx_size (items_per_task={args.items_per_task}, dots = individual seed runs)")
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(out_dir / "variation_check.png", dpi=150)
    log(f"wrote {out_dir / 'variation_check.png'} -- all done")


if __name__ == "__main__":
    main()
