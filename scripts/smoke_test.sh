#!/usr/bin/env bash
# One-shot Slurm smoke test: submits a tiny, fast pretrain job via
# slurm_pretrain.sh to confirm the pipeline (pixi env, rustler sampler build,
# torchrun launch, checkpointing) works end-to-end on this cluster.
#
# This is infra validation only -- 20 steps on a 2-block/d_model=64 model,
# not a real training run. Nothing here feeds sprint research output.
#
# Usage (from the relational-transformer root):
#   ACCOUNT=<acct> PARTITION=<part> QOS=<qos> ./scripts/smoke_test.sh
#
# Or hardcode ACCOUNT/PARTITION/QOS below once you know your cluster's
# values, so it becomes a true zero-argument `./scripts/smoke_test.sh`.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

ACCOUNT="${ACCOUNT:?set ACCOUNT to your Slurm account -- see: sacctmgr -p show associations user=\$USER}"
PARTITION="${PARTITION:?set PARTITION -- see: sinfo -o \"%P %G\"}"
QOS="${QOS:?set QOS -- see: sacctmgr -p show associations user=\$USER}"
GRES="${GRES:-gpu:1}"
MEM_PER_GPU="${MEM_PER_GPU:-16G}"

PRE_DIR=stanford-star/relbench-preprocessed \
VAL_PRE_DIR=stanford-star/relbench-preprocessed \
OUT_DIR="$HOME/ckpts/smoketest" \
GPUS_PER_NODE=1 \
EXTRA_ARGS="--total-steps 20 --eval-freq 1000 --num-blocks 2 --d-model 64 --num-heads 2 --d-ff 128 --no-compile" \
sbatch --nodes=1 --gres="$GRES" --time=00:15:00 \
       --account="$ACCOUNT" --partition="$PARTITION" --qos="$QOS" \
       --mem-per-gpu="$MEM_PER_GPU" \
       scripts/slurm_pretrain.sh
