#!/usr/bin/env python
"""K-means context reduction vs. raw truncation, at matched ctx_size (Sprint 02 / D-005).

Same seed-repeated sweep as eval_variance_check.py, but at each target ctx_size and each
shuffle_seed, runs *three* conditions and records all of them:

  - "baseline": the existing eval_variance_check.py condition -- sample directly at target
    ctx_size, evaluate as-is.
  - "diverse": oversample at oversample_factor * target ctx_size (default 5x), fit k-means
    over each non-target cell's own encoder embedding (n_slots = ctx_size - 1 clusters), and
    take one *medoid* per cluster -- guarantees every cluster contributes exactly one pick
    (maximal diversity, minimal redundancy). This was the original D-005 prototype's only
    k-means condition.
  - "packed": the SAME k-means fit as "diverse" (same oversampled pool, same clustering, same
    seed) -- but instead of one-per-cluster, rank every candidate by distance to its OWN
    cluster's centroid and take the top (ctx_size - 1) globally. A tight/dense cluster can
    contribute several picks; a loose cluster may contribute none. Same clusters, different
    take-rule -- tests whether enforcing diversity across clusters actually helps over just
    taking the globally most "typical" points.

"diverse" and "packed" share one oversampled Evaluator/batch and one k-means fit per
(ctx_size, seed) -- see rt.context_select.kmeans_select_multi -- so this only costs one extra
net.predict() call over the original 2-condition version, not a second full sampling pass.

The k-means step only needs the model's per-cell encoder (no attention), so the
expensive/quadratic transformer forward only ever runs at target ctx_size -- oversampling
by 5x does not reopen the ctx_size>=768 OOM boundary found in Sprint 01.

Local smoke test (tiny + eager, to catch bugs fast without waiting on Slurm):
    pixi run python scripts/eval_kmeans_check.py \\
        --target-ctx-sizes 128 --items-per-task 1 --n-seeds 2 --disable-compile \\
        --out-dir /tmp/kmeans_test

Cluster run (mig-preempt, via scripts/eval_kmeans_check.sh -- do NOT pass
--disable-compile there, same reasoning as eval_variance_check.sh):
    pixi run python scripts/eval_kmeans_check.py \\
        --target-ctx-sizes 128 192 256 384 512 --items-per-task 16 --n-seeds 8

LOGGING: same convention as eval_variance_check.py -- every log() line is
timestamped + elapsed-since-start and flush=True'd immediately.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # see eval_ctx_sweep.py -- same
# macOS OpenMP double-init fix, harmless no-op elsewhere.

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # see
# eval_ctx_sweep.py -- same fragmentation fix, harmless when it isn't needed.

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless -- this runs under Slurm / no display, always save to a file
import matplotlib.pyplot as plt
import torch
import torch._dynamo  # module level, not inside main() -- see eval_ctx_sweep.py's FIX note

# model.py compiles self.forward with torch.compile(dynamic=False): every distinct input
# *shape* gets its own specialized compiled kernel, no shape polymorphism. This script calls
# self() at up to 2 * len(target_ctx_sizes) distinct (batch_size, seq_len) shapes -- "baseline"
# and the shared "diverse"/"packed" oversampled Evaluator use different eval_bs at the same
# ctx_size, since eval_bs = tokens_per_gpu // sampled_ctx_size and they sample at different
# sizes (ctx_size vs. oversample_factor * ctx_size); "diverse" and "packed" themselves share
# one Evaluator so they don't add a third shape. torch._dynamo.config.recompile_limit defaults
# to 8: past that many distinct shapes for one guarded function, dynamo silently falls back to
# eager execution instead of compiling a 9th specialization. Eager/unfused flex_attention
# materializes the full (bs, heads, seq, seq) score matrix, which is what actually OOM'd on a
# real cluster run of the original 2-condition version (recompile_limit hit right at shape #9)
# -- not a host-RAM leak like the earlier eval_variance_check.py OOM. Raise the cap so every
# shape this script can produce gets its own compiled kernel instead of falling back.
torch._dynamo.config.recompile_limit = 64

from rt.checkpoints import load_rt_model
from rt.context_select import gather_along_seq, kmeans_select_multi
from rt.recipes import get_tasks
from rt.eval_utils import build_evaluator, metric_for

_T0 = time.monotonic()

MODES = ("diverse", "packed")


def log(msg: str) -> None:
    elapsed = time.monotonic() - _T0
    print(f"[{datetime.now().strftime('%H:%M:%S')} +{elapsed:7.1f}s] {msg}", flush=True)


def run_baseline(net, tasks, pre_dir, config, ctx_size, seed, args, device):
    """"baseline" condition: identical to eval_variance_check.py -- sample directly at
    ctx_size, evaluate as-is via the normal Evaluator.evaluate_raw path."""
    ev = build_evaluator(tasks, pre_dir, embedding_model=config["embedding_model"],
                         d_text=config["d_text"], device=device, ctx_size=ctx_size,
                         local_ctx_size=min(args.local_ctx_size_cap, ctx_size),
                         bfs_width=args.bfs_width, items_per_task=args.items_per_task,
                         shuffle_seed=seed, num_workers=0)
    metric_name = metric_value = n = None
    for task, _ctx, labels, preds_by_prefix, _nl in ev.evaluate_raw([(net, "")], [ctx_size]):
        metric_name, metric_value = metric_for(task.task_type, labels, preds_by_prefix[""],
                                                reg_metric=args.reg_metric)
        n = len(labels)
    del ev
    gc.collect()
    return metric_name, metric_value, n


def run_kmeans_variants(net, tasks, pre_dir, config, ctx_size, seed, args, device):
    """"diverse" and "packed" conditions: sample once at oversample_factor * ctx_size, fit
    k-means once per batch (rt.context_select.kmeans_select_multi), derive both modes' reduced
    batches from that single fit, evaluate each. Bypasses Evaluator.evaluate_raw's own
    truncation-to-ctx_size (net.predict's v[:, :ctx_size]) since the reduction here is
    kmeans_select_multi's job, not a prefix cut.

    Returns {mode: (metric_name, metric_value, n)} for mode in MODES.
    """
    oversample_ctx_size = args.oversample_factor * ctx_size
    ev = build_evaluator(tasks, pre_dir, embedding_model=config["embedding_model"],
                         d_text=config["d_text"], device=device, ctx_size=oversample_ctx_size,
                         local_ctx_size=min(args.local_ctx_size_cap, oversample_ctx_size),
                         bfs_width=args.bfs_width, items_per_task=args.items_per_task,
                         shuffle_seed=seed, num_workers=0)
    task = ev.tasks[0]
    eval_loader = ev.eval_loaders[task]
    eval_loader_iter = ev.eval_loader_iters[task]

    # Same n_batches formula as Evaluator.evaluate_raw (evaluator.py) -- cross-rank
    # uniformity doesn't matter here (world_size=1, ddp=False), just reproducing how
    # many batches items_per_task actually needs.
    n_batches = len(eval_loader.dataset)
    if ev.items_per_task is not None:
        n_batches = min(n_batches, max(1, ev.items_per_task // ev.eval_bs // ev.world_size))

    val_key = "boolean_values" if task.task_type == "clf" and not ev.bool_as_num else "number_values"

    labels_list = {m: [] for m in MODES}
    preds_list = {m: [] for m in MODES}
    net.eval()
    with torch.inference_mode():
        for _ in range(n_batches):
            batch = next(eval_loader_iter)
            batch_mask = batch.pop("batch_mask")

            keep_idx_by_mode = kmeans_select_multi(batch, net, k=ctx_size, seed=seed, modes=MODES)
            for m in MODES:
                reduced = gather_along_seq(batch, keep_idx_by_mode[m])
                preds_by_ctx = net.predict(reduced, [ctx_size], device, task,
                                           bool_as_num=ev.bool_as_num)
                yhat = preds_by_ctx[ctx_size].float().cpu()  # bfloat16 has no numpy equivalent

                y = (reduced[val_key].squeeze(-1).float()
                     * reduced["is_targets"].to(torch.float32)).sum(dim=1)

                labels_list[m].append(y[batch_mask])
                preds_list[m].append(yhat[batch_mask])

    out = {}
    for m in MODES:
        labels_np = torch.cat(labels_list[m]).numpy()
        preds_np = torch.cat(preds_list[m]).numpy()
        metric_name, metric_value = metric_for(task.task_type, labels_np, preds_np,
                                                reg_metric=args.reg_metric)
        out[m] = (metric_name, metric_value, len(labels_np))

    ev.eval_loader_iters[task] = iter(eval_loader)  # tidy, matches Evaluator's own re-prime
    del ev, eval_loader_iter
    gc.collect()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="stanford-star/rt-j/regression")
    ap.add_argument("--pre-dir", default="stanford-star/relbench-preprocessed")
    ap.add_argument("--task", default="rel-f1/driver-position",
                    help="'db/task-table' selector, as in scripts/eval.py --tasks")
    ap.add_argument("--target-ctx-sizes", type=int, nargs="+", default=[128, 192, 256, 384, 512])
    ap.add_argument("--oversample-factor", type=int, default=5,
                    help="'diverse'/'packed' sample at oversample_factor * target ctx_size, "
                         "then k-means-reduce back down to the target ctx_size")
    ap.add_argument("--items-per-task", type=int, default=16,
                    help="fixed rows sampled per run -- same subset (via shuffle_seed) for "
                         "all three conditions at a given seed")
    ap.add_argument("--n-seeds", type=int, default=8,
                    help="repeats per ctx_size, each with a different shuffle_seed (also used "
                         "as the k-means random_state for that repeat)")
    ap.add_argument("--bfs-width", type=int, default=32)
    ap.add_argument("--local-ctx-size-cap", type=int, default=256,
                    help="local_ctx_size = min(this, ctx_size) for each point, applied to "
                         "whichever ctx_size each condition actually samples at")
    ap.add_argument("--reg-metric", default="mae", choices=["mae", "r2"])
    ap.add_argument("--out-dir", default="cluster_run/kmeans_check")
    ap.add_argument("--disable-compile", action="store_true",
                    help="fall back to eager execution -- only pass this where the compiler "
                         "is broken (e.g. this Mac); leave it off on the cluster")
    args = ap.parse_args()
    log(f"starting: target_ctx_sizes={args.target_ctx_sizes} oversample_factor={args.oversample_factor} "
        f"items_per_task={args.items_per_task} n_seeds={args.n_seeds} task={args.task} "
        f"disable_compile={args.disable_compile}")

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
    # reasoning as eval_variance_check.py -- a crash partway through never loses what's
    # already computed.
    results = []
    n_runs = len(args.target_ctx_sizes) * args.n_seeds
    run_i = 0
    for ctx_size in args.target_ctx_sizes:
        oversample_ctx_size = args.oversample_factor * ctx_size
        for seed in range(args.n_seeds):
            run_i += 1
            log(f"[{run_i}/{n_runs}] ctx_size={ctx_size} shuffle_seed={seed} -- baseline ...")
            metric_name, baseline_value, baseline_n = run_baseline(
                net, tasks, args.pre_dir, config, ctx_size, seed, args, device)
            log(f"[{run_i}/{n_runs}] ctx_size={ctx_size} shuffle_seed={seed} -- "
                f"baseline {metric_name}={baseline_value:.4f} (n={baseline_n}) -- "
                f"diverse+packed (oversample={oversample_ctx_size}) ...")
            variants = run_kmeans_variants(
                net, tasks, args.pre_dir, config, ctx_size, seed, args, device)
            _mn, diverse_value, diverse_n = variants["diverse"]
            _mn, packed_value, packed_n = variants["packed"]
            log(f"[{run_i}/{n_runs}] ctx_size={ctx_size} shuffle_seed={seed}: "
                f"baseline {metric_name}={baseline_value:.4f} (n={baseline_n})  "
                f"diverse {metric_name}={diverse_value:.4f} (n={diverse_n})  "
                f"packed {metric_name}={packed_value:.4f} (n={packed_n}) -- done")
            results.append({
                "ctx_size": ctx_size, "oversample_ctx_size": oversample_ctx_size,
                "shuffle_seed": seed, "metric": metric_name,
                "baseline_value": baseline_value, "baseline_n": baseline_n,
                "diverse_value": diverse_value, "diverse_n": diverse_n,
                "packed_value": packed_value, "packed_n": packed_n,
            })
            (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    log(f"wrote {out_dir / 'results.json'}")

    # Plot: 3 grouped bars per ctx_size -- baseline (raw truncation), diverse (one medoid per
    # cluster), packed (same clustering, top-N globally by distance to own centroid) -- mean
    # +- 1 std across n_seeds, individual seed values overlaid as dots.
    metric_name = results[0]["metric"]
    x_pos = np.arange(len(args.target_ctx_sizes))
    width = 0.25
    series = {
        "baseline": ("#2b6cb0", "#1a365d", "baseline (raw truncation to ctx_size)", -1),
        "diverse":  ("#c53030", "#742a2a", "diverse (one medoid per cluster)", 0),
        "packed":   ("#2f855a", "#1c4532", "packed (same clusters, top-N by dist-to-centroid)", 1),
    }
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for mode, (bar_color, dot_color, label, offset) in series.items():
        means, stds = [], []
        for ctx_size in args.target_ctx_sizes:
            vals = [r[f"{mode}_value"] for r in results if r["ctx_size"] == ctx_size]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        ax.bar(x_pos + offset * width, means, width, yerr=stds, capsize=4,
              color=bar_color, label=label)
        for i, ctx_size in enumerate(args.target_ctx_sizes):
            vals = [r[f"{mode}_value"] for r in results if r["ctx_size"] == ctx_size]
            ax.scatter([x_pos[i] + offset * width] * len(vals), vals,
                      color=dot_color, zorder=3, s=16)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(args.target_ctx_sizes)
    ax.set_xlabel("ctx_size")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{args.task}: {metric_name}, baseline vs. diverse vs. packed "
                f"(from {args.oversample_factor}x oversampled pool)\n"
                f"mean ± 1 std across {args.n_seeds} seeds, items_per_task={args.items_per_task} "
                f"(dots = individual seed runs)")
    ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "kmeans_check.png", dpi=150)
    log(f"wrote {out_dir / 'kmeans_check.png'} -- all done")


if __name__ == "__main__":
    main()
