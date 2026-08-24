#!/usr/bin/env bash
# One-file sbatch wrapper for Sprint 01 / 1.4 Part B: submits scripts/eval_ctx_sweep.py to
# Slurm as a single job (sbatch --wrap -- no separate generic batch script needed).
#
# Defaults to a SMALL first-pass sweep on purpose (2 ctx_sizes, 8 items each, 15 min time
# limit) -- same philosophy as smoke_test.sh: confirm it actually runs and finishes fast on
# this cluster before committing to the full range. Raise CTX_SIZES/ITEMS_PER_TASK/TIME once
# you've seen a small run complete.
#
# Deliberately does NOT pass --disable-compile to eval_ctx_sweep.py: this runs on a real
# compute node with a native compiler, so torch.compile should actually produce the fast
# fused kernel (unlike the local Mac workaround -- see eval_ctx_sweep.py's docstring).
#
# --- test locally first (see eval_ctx_sweep.py's docstring) ---
#   pixi run python scripts/eval_ctx_sweep.py \
#       --ctx-sizes 128 --items-per-task 1 --disable-compile --out-dir /tmp/eval_sweep_test
#
# --- launch on the cluster (small first pass), once the local test runs clean ---
#   ACCOUNT=<acct> PARTITION=<part> QOS=<qos> ./scripts/eval_ctx_sweep.sh
#
# --- once that small run completes, the real sweep ---
#   ACCOUNT=<acct> PARTITION=<part> QOS=<qos> \
#   CTX_SIZES="128 256 512 1024 2048 4096 8192" ITEMS_PER_TASK=64 TIME=01:00:00 \
#   ./scripts/eval_ctx_sweep.sh
#
# Or hardcode ACCOUNT/PARTITION/QOS below once you know your cluster's values, so it becomes
# a true zero-argument `./scripts/eval_ctx_sweep.sh`.
#
# --- knobs (env vars) ---
#   CTX_SIZES        space-separated ctx_size values to sweep    [default: 128 256]
#   ITEMS_PER_TASK   rows sampled per ctx_size                   [default: 8]
#   CHECKPOINT       Hub model repo or local path                [default: stanford-star/rt-j/regression]
#   PRE_DIR          preprocessed data (Hub repo or local path)  [default: stanford-star/relbench-preprocessed]
#   TASK             'db/task-table' selector                    [default: rel-f1/driver-position]
#   OUT_DIR          where results.json / ctx_sweep.png land     [default: eval_sweep_out]
#   GRES             Slurm --gres value                          [default: gpu:1]
#   TIME             Slurm --time value                          [default: 00:15:00]
#   CONSTRAINT       Slurm --constraint (node feature) value      [default: unset -- no constraint]
#
# On a cluster where the same partition mixes GPU generations under one undifferentiated GRES
# name (`gpu:N`, no type), CONSTRAINT is how you pin a specific one via node features from
# `sinfo -N -o "%N %P %G %f"` -- e.g. this repo's cluster mixes H100 (`genoa,h100`, CC 9.0) and
# V100S (`rome,v100s`, CC 7.0) nodes under one `gpu` partition; the installed torch build only
# has kernels for CC>=7.5, so a plain `GRES=gpu:1` can land you on an unsupported V100S node.
# Two ways around that on this specific cluster (adjust to whatever your `sinfo -N` shows):
#   PARTITION=mig-preempt GRES=gpu:1g.10gb:1                 # MIG slice, same platform as the H100s
#   PARTITION=gpu GRES=gpu:1 CONSTRAINT=h100                 # full H100, explicitly avoids v100s
#
# --- watching it live ---
#   squeue -u $USER                              # confirm it's running (not still queued)
#   tail -f logs/eval-ctx-sweep-<jobid>.out       # real progress -- eval_ctx_sweep.py logs
#                                                  # timestamped lines with flush=True, and
#                                                  # PYTHONUNBUFFERED=1 below keeps stdout from
#                                                  # being batched up by Slurm's file redirect

set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

ACCOUNT="${ACCOUNT:?set ACCOUNT to your Slurm account -- see: sacctmgr -p show associations user=\$USER}"
PARTITION="${PARTITION:?set PARTITION -- see: sinfo -o \"%P %G\"}"
QOS="${QOS:?set QOS -- see: sacctmgr -p show associations user=\$USER}"
GRES="${GRES:-gpu:1}"
TIME="${TIME:-00:15:00}"
CONSTRAINT="${CONSTRAINT:-}"

CTX_SIZES="${CTX_SIZES:-128 256}"
ITEMS_PER_TASK="${ITEMS_PER_TASK:-8}"
CHECKPOINT="${CHECKPOINT:-stanford-star/rt-j/regression}"
PRE_DIR="${PRE_DIR:-stanford-star/relbench-preprocessed}"
TASK="${TASK:-rel-f1/driver-position}"
OUT_DIR="${OUT_DIR:-eval_sweep_out}"

echo "ctx_sizes=$CTX_SIZES items_per_task=$ITEMS_PER_TASK checkpoint=$CHECKPOINT out_dir=$OUT_DIR time=$TIME constraint=${CONSTRAINT:-<none>}"

# --constraint is only included when CONSTRAINT is set -- sbatch has no "no constraint" value,
# omitting the flag entirely is how you leave scheduling unconstrained.
CONSTRAINT_ARGS=()
if [ -n "$CONSTRAINT" ]; then
    CONSTRAINT_ARGS=(--constraint="$CONSTRAINT")
fi

sbatch --job-name=eval-ctx-sweep --output=logs/eval-ctx-sweep-%j.out \
       --nodes=1 --gres="$GRES" --time="$TIME" \
       --account="$ACCOUNT" --partition="$PARTITION" --qos="$QOS" \
       "${CONSTRAINT_ARGS[@]}" \
       --wrap="export PYTHONUNBUFFERED=1; pixi run build-sampler >/dev/null 2>&1 || true; pixi run python scripts/eval_ctx_sweep.py --checkpoint '$CHECKPOINT' --pre-dir '$PRE_DIR' --task '$TASK' --ctx-sizes $CTX_SIZES --items-per-task $ITEMS_PER_TASK --out-dir '$OUT_DIR'"
