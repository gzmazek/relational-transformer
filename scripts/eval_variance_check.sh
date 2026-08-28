#!/usr/bin/env bash
# One-file sbatch wrapper for Sprint 01 / 1.4 follow-up: submits
# scripts/eval_variance_check.py to Slurm as a single job.
#
# Meant for the mig-preempt partition -- ctx_sizes up to 512 (the default set)
# already fit its 10GB VRAM without OOM (confirmed in the ctx_size sweep), so
# this is the fast/cheap setup: short queue wait, no need for a full H100.
#
# Deliberately does NOT pass --disable-compile: this runs on a real compute
# node with a native compiler (see eval_ctx_sweep.sh for the same reasoning).
#
# --- test locally first (see eval_variance_check.py's docstring) ---
#   pixi run python scripts/eval_variance_check.py \
#       --ctx-sizes 128 --items-per-task 1 --n-seeds 2 --disable-compile \
#       --out-dir /tmp/variance_test
#
# --- launch on the cluster (small first pass -- genuinely smaller than the real
# --- sweep below, and a different OUT_DIR so it can never collide with it even if
# --- both end up running at once) ---
#   ACCOUNT=<acct> PARTITION=mig-preempt QOS=<qos> GRES=gpu:1g.10gb:1 \
#   CTX_SIZES="128 256" ITEMS_PER_TASK=4 N_SEEDS=2 TIME=00:15:00 \
#   OUT_DIR=cluster_run/variation_check_smoke \
#   ./scripts/eval_variance_check.sh
#
# --- once that completes, the real sweep (this is just the tool's defaults --
# --- CTX_SIZES/ITEMS_PER_TASK/N_SEEDS below don't need restating) ---
#   ACCOUNT=<acct> PARTITION=mig-preempt QOS=<qos> GRES=gpu:1g.10gb:1 \
#   ./scripts/eval_variance_check.sh
#
# --- knobs (env vars) ---
#   CTX_SIZES         space-separated ctx_size values to sweep    [default: 128 192 256 384 512]
#   ITEMS_PER_TASK    fixed rows sampled per run (not swept)      [default: 16]
#   N_SEEDS           repeats per ctx_size                        [default: 8]
#   CHECKPOINT        Hub model repo or local path                [default: stanford-star/rt-j/regression]
#   PRE_DIR           preprocessed data (Hub repo or local path)  [default: stanford-star/relbench-preprocessed]
#   TASK              'db/task-table' selector                    [default: rel-f1/driver-position]
#   OUT_DIR           where results.json / variation_check.png land [default: cluster_run/variation_check]
#   GRES              Slurm --gres value                          [default: gpu:1]
#   TIME              Slurm --time value                          [default: 00:30:00]
#   CONSTRAINT        Slurm --constraint (node feature) value      [default: unset -- no constraint]
#
# --- watching it live ---
#   squeue -u $USER
#   tail -f logs/eval-variance-check-<jobid>.out

set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

ACCOUNT="${ACCOUNT:?set ACCOUNT to your Slurm account -- see: sacctmgr -p show associations user=\$USER}"
PARTITION="${PARTITION:?set PARTITION -- see: sinfo -o \"%P %G\"}"
QOS="${QOS:?set QOS -- see: sacctmgr -p show associations user=\$USER}"
GRES="${GRES:-gpu:1}"
TIME="${TIME:-00:30:00}"
CONSTRAINT="${CONSTRAINT:-}"

CTX_SIZES="${CTX_SIZES:-128 192 256 384 512}"
ITEMS_PER_TASK="${ITEMS_PER_TASK:-16}"
N_SEEDS="${N_SEEDS:-8}"
CHECKPOINT="${CHECKPOINT:-stanford-star/rt-j/regression}"
PRE_DIR="${PRE_DIR:-stanford-star/relbench-preprocessed}"
TASK="${TASK:-rel-f1/driver-position}"
OUT_DIR="${OUT_DIR:-cluster_run/variation_check}"

echo "ctx_sizes=$CTX_SIZES items_per_task=$ITEMS_PER_TASK n_seeds=$N_SEEDS checkpoint=$CHECKPOINT out_dir=$OUT_DIR time=$TIME constraint=${CONSTRAINT:-<none>}"

CONSTRAINT_ARGS=()
if [ -n "$CONSTRAINT" ]; then
    CONSTRAINT_ARGS=(--constraint="$CONSTRAINT")
fi

sbatch --job-name=eval-variance-check --output=logs/eval-variance-check-%j.out \
       --nodes=1 --gres="$GRES" --time="$TIME" \
       --account="$ACCOUNT" --partition="$PARTITION" --qos="$QOS" \
       "${CONSTRAINT_ARGS[@]}" \
       --wrap="export PYTHONUNBUFFERED=1; pixi run build-sampler >/dev/null 2>&1 || true; pixi run python scripts/eval_variance_check.py --checkpoint '$CHECKPOINT' --pre-dir '$PRE_DIR' --task '$TASK' --ctx-sizes $CTX_SIZES --items-per-task $ITEMS_PER_TASK --n-seeds $N_SEEDS --out-dir '$OUT_DIR'"
