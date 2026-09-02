"""Pluggable context-reduction hook (Sprint 02 / D-005 prototype).

Sits between a batch sampled at a larger ("oversampled") ctx_size and the model's
real forward pass. The intended call shape, mirroring the sketch in Sprint 01's
D-005 idea block::

    raw = sampler.batch_py(idx, bs, oversample_ctx_size)         # existing API
    batch = process_batch(raw, d_text, bool_as_num)               # existing, unchanged
    keep_idx = kmeans_select(batch, net, k=target_ctx_size)        # this module
    reduced = gather_along_seq(batch, keep_idx)                    # this module
    preds = net.predict(reduced, [target_ctx_size], device, task, bool_as_num=True)

``embed_cells`` and ``kmeans_select`` only need the model's per-cell *encoder*
(``enc_dict``/``norm_dict``, model.py's pre-``self.blocks`` step) -- no attention,
so they're cheap even at a large oversampled seq_len. The real (quadratic-cost)
transformer forward only ever runs on the *reduced* batch, at ``target_ctx_size``.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import KMeans

VALUE_TYPES = ["number", "text", "datetime", "boolean"]


def embed_cells(batch, net):
    """Per-cell embedding ``x``, computed exactly as ``RelationalTransformer.forward``
    computes it before running ``self.blocks`` (model.py's ``enc_dict``/``norm_dict``
    step): the ``col_name`` embedding plus the cell's own typed-value embedding
    (number/text/datetime/boolean, selected by ``sem_types``), each through the
    model's own ``Linear`` + ``RMSNorm``. This is the literal vector the model adds
    into its context sequence -- reproduced standalone here (no grad, no attention)
    so k-means can select on it before the reduced batch ever reaches ``self.blocks``.

    Returns ``(bs, seq_len, d_model)``, on ``net``'s device/dtype. Target and padding
    cells are left at 0 (``forward`` instead adds ``mask_embs`` for the target and
    nothing for padding -- irrelevant here since neither is a k-means candidate).
    """
    device = next(net.parameters()).device
    dtype = net.enc_dict["number"].weight.dtype

    with torch.no_grad():
        col_name_values = batch["col_name_values"].to(device=device, dtype=dtype)
        x = net.norm_dict["col_name"](net.enc_dict["col_name"](col_name_values))

        sem_types = batch["sem_types"].to(device)
        is_targets = batch["is_targets"].to(device)
        is_padding = batch["is_padding"].to(device)
        for i, t in enumerate(VALUE_TYPES):
            v = batch[f"{t}_values"].to(device=device, dtype=dtype)
            v = torch.where(torch.isnan(v), torch.zeros_like(v), v)  # match forward()'s NaN guard
            mask = (sem_types == i) & ~is_targets & ~is_padding
            x = x + net.norm_dict[t](net.enc_dict[t](v)) * mask[..., None]
    return x


def gather_along_seq(batch, keep_idx):
    """Reduce every per-cell tensor in ``batch`` (shape ``(bs, seq_len, ...)``) to the
    ``keep_idx`` positions (``(bs, k)``) along the seq dim. Row-level tensors (e.g.
    ``batch_mask``, shape ``(bs,)``) pass through unchanged."""
    keep_idx = keep_idx.to(next(iter(batch.values())).device)
    out = {}
    for key, v in batch.items():
        if v.dim() < 2:
            out[key] = v
            continue
        idx = keep_idx.view(*keep_idx.shape, *([1] * (v.dim() - 2)))
        out[key] = v.gather(1, idx.expand(-1, -1, *v.shape[2:]))
    return out


def kmeans_select_multi(batch, net, k, seed=0, modes=("diverse", "packed")):
    """Fit k-means *once* per row -- ``n_slots = k - 1`` clusters over the non-target,
    non-padding candidates, same as the original D-005 prototype -- and derive
    ``keep_idx`` for every mode in ``modes`` from that single fit, so different
    take-rules are compared against the exact same clustering rather than
    accidentally-identical separate fits (both modes see the same oversampled
    candidate pool and the same ``KMeans(..., random_state=seed)`` result):

    - ``"diverse"`` (the original D-005 prototype): one *medoid* per cluster -- the
      real candidate closest to its own cluster's centroid -- one per cluster,
      guaranteeing every cluster contributes exactly one pick. Real observed cells,
      not synthetic centroids, same "real rows" spirit as CUR (D-002).
    - ``"packed"``: rank ALL candidates by distance to their OWN cluster's centroid
      and take the top ``n_slots`` globally -- a tight/dense cluster can contribute
      several picks, a loose cluster may contribute none. Same clustering as
      ``"diverse"``, a different take-rule from it.

    The target cell always fills 1 of the ``k`` slots, in every mode. A row with
    fewer than ``k - 1`` real candidates skips clustering (nothing to compare) and
    just keeps all of it, padding the rest, same as the model already handles
    padding. Phantom rows (no target cell, from eval batch overshoot -- see
    ``net.predict``'s docstring) are cheap to detect and skip clustering for too,
    since their result is discarded downstream via ``batch_mask``.

    Returns ``{mode: keep_idx}``, each ``(bs, k)`` ``LongTensor`` of seq-dim indices
    into ``batch``.
    """
    x = embed_cells(batch, net)  # (bs, seq_len, d_model)
    bs, seq_len, _ = x.shape
    is_targets = batch["is_targets"].to(x.device)
    is_padding = batch["is_padding"].to(x.device)
    is_candidate = ~is_targets & ~is_padding

    keep_idx = {m: torch.zeros(bs, k, dtype=torch.long) for m in modes}
    for row in range(bs):
        target_idx = is_targets[row].nonzero(as_tuple=True)[0]  # 1 for a real row, 0 for phantom
        n_slots = k - target_idx.numel()

        if target_idx.numel() == 0:
            # phantom row (no target) -- result is discarded via batch_mask downstream,
            # skip clustering and just fill with whatever's there. Same for every mode.
            row_idx = torch.arange(min(k, seq_len), device=x.device)
            if row_idx.numel() < k:
                top_up = torch.arange(seq_len, device=x.device)[: k - row_idx.numel()]
                row_idx = torch.cat([row_idx, top_up])
            for m in modes:
                keep_idx[m][row] = row_idx[:k].cpu()
            continue

        cand_idx = is_candidate[row].nonzero(as_tuple=True)[0]
        pad_idx = is_padding[row].nonzero(as_tuple=True)[0]

        if cand_idx.numel() <= n_slots:
            # not enough real candidates to cluster -- keep them all, pad the rest.
            # No selection choice to make, so every mode gets the same result.
            n_pad = n_slots - cand_idx.numel()
            fill = pad_idx[:n_pad]
            row_idx = torch.cat([target_idx, cand_idx, fill])
            if row_idx.numel() < k:
                top_up = torch.arange(seq_len, device=x.device)[: k - row_idx.numel()]
                row_idx = torch.cat([row_idx, top_up])
            for m in modes:
                keep_idx[m][row] = row_idx[:k].cpu()
            continue

        x_cand = x[row, cand_idx].float().cpu().numpy()
        km = KMeans(n_clusters=n_slots, n_init="auto", random_state=seed).fit(x_cand)
        labels = km.labels_
        d_own = ((x_cand - km.cluster_centers_[labels]) ** 2).sum(axis=1)  # dist to OWN centroid

        for m in modes:
            if m == "diverse":
                picks_local = np.array([
                    (members := (labels == c).nonzero()[0])[d_own[members].argmin()]
                    for c in range(n_slots)
                ])
            elif m == "packed":
                picks_local = np.argsort(d_own)[:n_slots]
            else:
                raise ValueError(f"unknown mode {m!r}")
            picks = cand_idx[torch.as_tensor(picks_local, dtype=torch.long, device=x.device)]
            row_idx = torch.cat([target_idx, picks])
            keep_idx[m][row] = row_idx[:k].cpu()

    return keep_idx


def kmeans_select(batch, net, k, seed=0, mode="diverse"):
    """Single-mode convenience wrapper around :func:`kmeans_select_multi` (see its
    docstring for the exact selection rule per mode)."""
    return kmeans_select_multi(batch, net, k, seed=seed, modes=(mode,))[mode]
