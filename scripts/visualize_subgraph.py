#!/usr/bin/env python
"""Visualize one sampled context as a cell-level graph (Sprint 02 / 2.1 follow-up).

Cell-level, not row-level: each *cell* (a specific row x column pair) is its own node here --
matching exactly what RelationalTransformer.forward() attends over, not the row/entity-level
view the ctx-viz tool (Sprint 01 / 1.1) shows. Edges reproduce model.py's own three
attention-mask definitions verbatim (model.py lines ~391-403):

    same_node    = node_idxs[:,:,None] == node_idxs[:,None,:]                       "feat" (part 1)
    kv_in_f2p    = (node_idxs[:,None,:,None] == f2p_nbr_idxs[:,:,None,:]).any(-1)   "feat" (part 2) / "nbr" (transpose)
    same_col_tbl = (col_name_idxs[:,:,None]==col_name_idxs[:,None,:])
                 & (table_name_idxs[:,:,None]==table_name_idxs[:,None,:])           "col"

so this shows literally the structure the model computes, not an approximation of it -- see
Sprint 02's notebook, subtask 2.1, for the derivation.

No model forward pass needed -- this only samples a batch and inspects its own index tensors
(no attention, no compile), so it's fast even on CPU/eager.

Usage:
    pixi run python scripts/visualize_subgraph.py --ctx-size 28 --out figures/subgraph.png
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from rt.checkpoints import load_rt_model
from rt.eval_utils import build_evaluator
from rt.recipes import get_tasks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="stanford-star/rt-j/regression")
    ap.add_argument("--pre-dir", default="stanford-star/relbench-preprocessed")
    ap.add_argument("--task", default="rel-f1/driver-position")
    ap.add_argument("--ctx-size", type=int, default=28,
                    help="small on purpose -- this is for legibility, the real sweeps use "
                         "128-2560; see Sprint 01's own ctx_size sweep for that range")
    ap.add_argument("--local-ctx-size", type=int, default=28)
    ap.add_argument("--bfs-width", type=int, default=32)
    ap.add_argument("--shuffle-seed", type=int, default=0)
    ap.add_argument("--out", default="figures/subgraph.png")
    args = ap.parse_args()

    # No net(...) call anywhere below -- we only need config (embedding_model/d_text) to
    # build the sampler, never the model weights, so this stays fast on CPU/eager.
    _net, config = load_rt_model(args.checkpoint, device="cpu", compile=False)

    all_tasks = get_tasks("relbench_eval_test", args.pre_dir)
    tasks = [t for t in all_tasks if f"{t.db_name}/{t.table_name}" == args.task]
    if not tasks:
        raise SystemExit(f"{args.task} not found in relbench_eval_test tasks for {args.pre_dir}")

    ev = build_evaluator(tasks, args.pre_dir, embedding_model=config["embedding_model"],
                         d_text=config["d_text"], device="cpu", ctx_size=args.ctx_size,
                         local_ctx_size=args.local_ctx_size, bfs_width=args.bfs_width,
                         items_per_task=1, shuffle_seed=args.shuffle_seed, num_workers=0)
    task = ev.tasks[0]
    batch = next(ev.eval_loader_iters[task])

    # Row 0 of the batch, real (non-padding) cells only.
    is_padding = batch["is_padding"][0].bool()
    real = (~is_padding).nonzero(as_tuple=True)[0]
    n = real.numel()
    print(f"sampled {n} real cells at ctx_size={args.ctx_size}")

    node_idxs = batch["node_idxs"][0, real]
    f2p_nbr_idxs = batch["f2p_nbr_idxs"][0, real]  # (n, 5), values are NODE idxs, not positions
    col_name_idxs = batch["col_name_idxs"][0, real]
    table_name_idxs = batch["table_name_idxs"][0, real]
    is_targets = batch["is_targets"][0, real]
    bfs_depths = batch["bfs_depths"][0, real]
    sem_types = batch["sem_types"][0, real]

    # Same three edge definitions as model.py's forward(), restricted to these n cells.
    same_node = node_idxs[:, None] == node_idxs[None, :]
    # f2p_nbr_idxs holds NODE indices -- a cell i's edge to cell j fires if j's node is
    # listed among i's (up to 5) f2p neighbors.
    kv_in_f2p = (node_idxs[None, :, None] == f2p_nbr_idxs[:, None, :]).any(-1)
    same_col_table = (col_name_idxs[:, None] == col_name_idxs[None, :]) & \
                     (table_name_idxs[:, None] == table_name_idxs[None, :])

    G = nx.Graph()
    for i in range(n):
        G.add_node(i, node_idx=int(node_idxs[i]), bfs_depth=int(bfs_depths[i]),
                   is_target=bool(is_targets[i]), sem_type=int(sem_types[i]))

    same_row_edges, fk_edges, col_edges = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            if same_node[i, j]:
                same_row_edges.append((i, j))
            elif kv_in_f2p[i, j] or kv_in_f2p[j, i]:  # either direction -> one FK edge
                fk_edges.append((i, j))
            elif same_col_table[i, j]:
                col_edges.append((i, j))
    print(f"edges: same-row={len(same_row_edges)} fk={len(fk_edges)} same-col-table={len(col_edges)}")

    # Layout: two-level. First lay out ROWS (node_idx) via a spring layout on the row-level
    # graph induced from the cell-level FK/same-col-table edges -- lets actual connectivity
    # decide row placement, rather than a rigid angular slot per bfs_depth (which degenerates
    # to a straight line whenever a depth has only one row, as most do in a small sample).
    # Then each row's own cells are placed in a small local circle around that row's center.
    node_depth = {}
    for i in range(n):
        nd = int(node_idxs[i])
        node_depth.setdefault(nd, int(bfs_depths[i]))

    row_graph = nx.Graph()
    row_graph.add_nodes_from(node_depth.keys())
    for i, j in fk_edges + col_edges:
        ni, nj = int(node_idxs[i]), int(node_idxs[j])
        if ni != nj:
            row_graph.add_edge(ni, nj)

    if row_graph.number_of_edges() > 0:
        row_center = nx.spring_layout(row_graph, seed=0, k=1.4 / max(len(node_depth) ** 0.5, 1))
    else:
        # no inter-row edges at all (shouldn't happen with a real f2p-connected sample, but
        # stay robust) -- fall back to a simple circle so rows don't all land on top of each other
        row_center = nx.circular_layout(row_graph)
    # Scale by bfs_depth so hop-distance from the target is still visually legible (further
    # rows pushed outward), on top of spring_layout's connectivity-driven placement.
    for nd in row_center:
        x, y = row_center[nd]
        scale = 1.0 + 1.6 * node_depth[nd]
        row_center[nd] = (x * scale, y * scale)

    cells_by_row: dict[int, list[int]] = {}
    for i in range(n):
        cells_by_row.setdefault(int(node_idxs[i]), []).append(i)

    pos = {}
    for nd, cells in cells_by_row.items():
        cx, cy = row_center[nd]
        m = len(cells)
        local_r = 0.22 if m > 1 else 0.0
        for k, i in enumerate(cells):
            phi = 2 * np.pi * k / max(m, 1)
            pos[i] = (cx + local_r * np.cos(phi), cy + local_r * np.sin(phi))

    fig, ax = plt.subplots(figsize=(9, 9))
    nx.draw_networkx_edges(G, pos, edgelist=same_row_edges, edge_color="#a0aec0", width=0.8,
                           alpha=0.6, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=col_edges, edge_color="#dd6b20", width=1.0,
                           style="dashed", alpha=0.7, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=fk_edges, edge_color="#2b6cb0", width=1.6,
                           alpha=0.85, ax=ax)

    node_colors = ["#c53030" if bool(is_targets[i]) else "#2f855a" for i in range(n)]
    node_sizes = [220 if bool(is_targets[i]) else 90 for i in range(n)]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                           edgecolors="black", linewidths=0.5, ax=ax)

    for nd, (cx, cy) in row_center.items():
        ax.text(cx, cy + 0.32, f"d={node_depth[nd]}", fontsize=6.5, color="#718096", ha="center")

    legend_elems = [
        plt.Line2D([0], [0], color="#a0aec0", lw=2, label="same row (same node_idx)"),
        plt.Line2D([0], [0], color="#2b6cb0", lw=2, label="FK neighbor (f2p / nbr)"),
        plt.Line2D([0], [0], color="#dd6b20", lw=2, linestyle="dashed", label="same column + table (Tier-1)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#c53030", markersize=10, label="target cell"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2f855a", markersize=8, label="other cell"),
    ]
    ax.legend(handles=legend_elems, loc="upper left", fontsize=8, frameon=False)
    ax.set_title(f"{args.task}: one sampled context, cell-level graph (ctx_size={args.ctx_size}, n={n} real cells)\n"
                f"each small cluster = one row's own cells; d=N labels = bfs_depth from target")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.autoscale_view()
    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
