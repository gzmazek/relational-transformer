#!/usr/bin/env python
"""Context size vs. model performance sweep (Sprint 01 / 1.4 Part B).

Loads a checkpoint, evaluates it at each --ctx-size on a small subsample of one
task, and plots the metric vs. context size. Same script for local testing and
a real cluster run -- only the arguments change:

Local smoke test (tiny + eager, to catch bugs fast without waiting on Slurm):
    pixi run python scripts/eval_ctx_sweep.py \\
        --ctx-sizes 128 --items-per-task 1 --disable-compile \\
        --out-dir /tmp/eval_sweep_test

Real cluster run (via scripts/eval_ctx_sweep.sh -- do NOT pass --disable-compile
there: the cluster has a working native compiler, so torch.compile should
actually produce the fast fused kernel instead of the slow eager fallback
this flag exists to work around):
    pixi run python scripts/eval_ctx_sweep.py \\
        --ctx-sizes 128 256 512 1024 2048 4096 8192 --items-per-task 64

See the Insight in experiment.ipynb (Sprint 01 / 1.4 Part B) for why
--disable-compile and the OpenMP fix exist -- both were found by actually
running this on macOS, not guessed.

LOGGING: every log() line is timestamped + elapsed-since-start and flush=True'd
immediately, so `tail -f logs/eval-ctx-sweep-<jobid>.out` on the cluster shows
real progress as it happens rather than a batch of output at the end. Python
buffers stdout when it isn't a terminal (i.e. redirected to Slurm's log file),
which would otherwise delay everything -- eval_ctx_sweep.sh also sets
PYTHONUNBUFFERED=1 as a second layer of the same fix.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # harmless no-op if there's no conflict;
# needed on macOS where torch and numpy/the rustler extension each bundle their own OpenMP
# runtime and loading both in one process aborts with "OMP: Error #15" otherwise.

import matplotlib
matplotlib.use("Agg")  # headless -- this runs under Slurm / no display, always save to a file
import matplotlib.pyplot as plt
import torch
import torch._dynamo  # imported at module level, not inside main() -- see the FIX note below

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
    ap.add_argument("--ctx-sizes", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--items-per-task", type=int, default=2,
                    help="rows sampled per ctx_size -- this is a debug/relative-comparison "
                         "metric on a small subsample, not a RelBench-leaderboard score")
    ap.add_argument("--bfs-width", type=int, default=32)
    ap.add_argument("--local-ctx-size-cap", type=int, default=256,
                    help="local_ctx_size = min(this, ctx_size) for each point")
    ap.add_argument("--reg-metric", default="mae", choices=["mae", "r2"])
    ap.add_argument("--out-dir", default="eval_sweep_out")
    ap.add_argument("--disable-compile", action="store_true",
                    help="fall back to eager execution instead of torch.compile -- only pass "
                         "this where the compiler is broken (e.g. this Mac); leave it off on "
                         "the cluster so the fast compiled kernel actually runs")
    args = ap.parse_args()
    log(f"starting: ctx_sizes={args.ctx_sizes} items_per_task={args.items_per_task} "
        f"task={args.task} checkpoint={args.checkpoint} disable_compile={args.disable_compile}")

    # FIX: torch._dynamo used to be imported right here, inside this `if`. `import a.b` binds
    # the top-level name `a` (not `a.b`) in the *current* scope -- so having it anywhere inside
    # main(), even in a branch that doesn't run, makes Python treat `torch` as local to the
    # whole function and shadow the module-level import. When --disable-compile wasn't passed
    # (the cluster's normal case), that branch never executed, so the local `torch` was never
    # assigned -- caught on a real cluster run: "UnboundLocalError: cannot access local
    # variable 'torch'" on the very next line. Moved the import to module level instead.
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

    # FIX (after a real cluster run: 128/256/512 succeeded, 1024 hit a fatal CUDA OOM on a
    # 10GB MIG slice, and the unhandled exception crashed the whole process -- losing the three
    # already-computed results too, since results.json/ctx_sweep.png were only written after the
    # full loop). OOM at the top of a ctx_size sweep is an expected outcome on a memory-limited
    # GPU, not a rare edge case -- catch it per ctx_size, keep going (a bigger ctx_size will just
    # OOM again quickly, not hang), and write results.json after *every* ctx_size so a later
    # crash can never lose earlier results again.
    results_by_ctx = []
    for i, ctx_size in enumerate(args.ctx_sizes, 1):
        local_ctx_size = min(args.local_ctx_size_cap, ctx_size)
        log(f"[{i}/{len(args.ctx_sizes)}] ctx_size={ctx_size} (local_ctx_size={local_ctx_size}) -- building evaluator ...")
        try:
            ev = build_evaluator(tasks, args.pre_dir, embedding_model=config["embedding_model"],
                                 d_text=config["d_text"], device=device, ctx_size=ctx_size,
                                 local_ctx_size=local_ctx_size, bfs_width=args.bfs_width,
                                 items_per_task=args.items_per_task, shuffle_seed=0)
            log(f"[{i}/{len(args.ctx_sizes)}] ctx_size={ctx_size} -- running forward pass over {args.items_per_task} row(s) ...")
            for task, _ctx, labels, preds_by_prefix, _nl in ev.evaluate_raw([(net, "")], [ctx_size]):
                metric_name, metric_value = metric_for(task.task_type, labels, preds_by_prefix[""],
                                                        reg_metric=args.reg_metric)
                n = len(labels)
            results_by_ctx.append({"ctx_size": ctx_size, "metric": metric_name, "value": metric_value, "n": n})
            log(f"[{i}/{len(args.ctx_sizes)}] ctx_size={ctx_size}: {metric_name}={metric_value:.4f} (n={n}) -- done")
        except torch.OutOfMemoryError as e:
            log(f"[{i}/{len(args.ctx_sizes)}] ctx_size={ctx_size} -- OUT OF MEMORY, skipping: {e}")
            results_by_ctx.append({"ctx_size": ctx_size, "metric": None, "value": None, "n": 0, "error": "OOM"})
            torch.cuda.empty_cache()
        (out_dir / "results.json").write_text(json.dumps(results_by_ctx, indent=2))

    log(f"wrote {out_dir / 'results.json'}")

    successful = [r for r in results_by_ctx if r["value"] is not None]
    oom_sizes = [r["ctx_size"] for r in results_by_ctx if r["value"] is None]
    if oom_sizes:
        log(f"skipped (OOM): {oom_sizes}")
    if not successful:
        log("no ctx_size completed successfully -- nothing to plot")
        return

    metric_name = successful[0]["metric"]
    xs = [r["ctx_size"] for r in successful]
    ys = [r["value"] for r in successful]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, ys, marker="o", color="#2b6cb0")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(xs)
    ax.set_xlabel("ctx_size")
    ax.set_ylabel(metric_name)
    better = "lower is better" if metric_name == "mae" else "higher is better"
    oom_note = f" -- OOM, skipped: {oom_sizes}" if oom_sizes else ""
    ax.set_title(f"{args.task}: {metric_name} vs. context size ({better})\n"
                 f"debug metric on a {args.items_per_task}-row local subsample, not a RelBench-leaderboard score{oom_note}")
    plt.tight_layout()
    fig.savefig(out_dir / "ctx_sweep.png", dpi=150)
    log(f"wrote {out_dir / 'ctx_sweep.png'} -- all done")


if __name__ == "__main__":
    main()
