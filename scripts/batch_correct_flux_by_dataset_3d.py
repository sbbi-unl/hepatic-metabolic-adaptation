#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
  python batch_correct_flux_by_dataset_3d.py \\
    --input combined_reaction_flux_comparison_extended.csv \\
    --output_csv combined_reaction_flux_comparison_extended_batch_corrected.csv \\
    --outdir batch_correction_outputs \\
    --interactive_3d

Optional:
  --sample_suffix "_Flux"
  --ignore_regex "(MeanFlux|StdFlux|SEMFlux)$"
  --dataset_regex "(?:GSE|GSM)\\d+"
  --group_regex "(?i)(SCD|HFD|KD|WD|LFD|NCD|CTRL|TREAT)"
  --no_plots       (skip all PNG plots)
  --no_3d          (skip the 3D PCA PNG plots; 2D plots still generated)
  --interactive_3d (also export interactive Plotly 3D HTML plots)
"""

import os
import re
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # needed for 3D projection

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Plotly optional import for interactive 3D
try:
    import plotly.graph_objects as go
    _PLOTLY_OK = True
except Exception:
    _PLOTLY_OK = False


# ----------------------------- CLI -------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Remove dataset batch effects from flux tables while preserving groups (now with 3D + interactive 3D PCA).")
    ap.add_argument("--input", "-i", required=True, help="Path to combined flux CSV.")
    ap.add_argument("--output_csv", "-o", required=True, help="Path to write batch-corrected CSV.")
    ap.add_argument("--outdir", default="batch_correction_outputs", help="Directory for plots & summary.")
    ap.add_argument("--sample_suffix", default="_Flux", help="Suffix used to identify sample columns.")
    ap.add_argument("--ignore_regex", default=r"(MeanFlux|StdFlux|SEMFlux)$",
                    help="Regex to drop aggregate columns from sample set.")
    ap.add_argument("--dataset_regex", default=r"(?:GSE|GSM)\d+",
                    help="Regex to parse dataset IDs from column names.")
    ap.add_argument("--group_regex", default=None,
                    help="Regex to parse group labels from column names. If omitted, uses prefix before first underscore.")
    ap.add_argument("--no_plots", action="store_true", help="Skip generating PNG plots (both 2D and 3D).")
    ap.add_argument("--no_3d", action="store_true", help="Skip generating 3D PCA PNG plots (keep 2D plots).")
    ap.add_argument("--interactive_3d", action="store_true", help="Also export interactive 3D PCA HTML plots (requires Plotly).")
    return ap.parse_args()


# -------------------------- Utilities ----------------------------------------

def detect_sample_columns(columns, sample_suffix, ignore_regex):
    ignore = re.compile(ignore_regex, re.IGNORECASE) if ignore_regex else None
    sample_cols = []
    for c in columns:
        s = str(c)
        if s.endswith(sample_suffix) and not (ignore and ignore.search(s)):
            sample_cols.append(s)
    return sample_cols


def parse_meta_from_col(col, dataset_regex, group_regex):
    col_str = str(col)

    # dataset
    ds = "UNKNOWN"
    if dataset_regex:
        m = re.search(dataset_regex, col_str, re.IGNORECASE)
        if m:
            ds = m.group(0).upper()

    # group
    if group_regex:
        mg = re.search(group_regex, col_str)
        group = mg.group(0) if mg else col_str.split("_", 1)[0]
    else:
        group = col_str.split("_", 1)[0]

    return ds, group


def pca_and_partition(X, meta, outdir, tag, make_plots=True, make_plots_3d=True):
    """
    Standardize X (samples x features), compute PCA, and estimate how much of PC1
    aligns with dataset vs group (simple ANOVA-like SS partition on group means).
    Also generate 2D and (optionally) 3D scatter plots.
    """
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    # ensure 3 comps if 3D plots are requested and feasible
    min_comps = 3 if (make_plots and make_plots_3d and Xs.shape[0] >= 3) else 2
    ncomp = max(min_comps, min(10, Xs.shape[0] - 1))
    pca = PCA(n_components=ncomp)
    Xp = pca.fit_transform(Xs)

    pc1 = Xp[:, 0]
    grand = pc1.mean()
    ds_levels = meta["dataset"].unique().tolist()
    gp_levels = meta["group"].unique().tolist()

    ss_ds = sum(((pc1[(meta["dataset"] == ds)].mean() - grand) ** 2) *
                (meta["dataset"] == ds).sum() for ds in ds_levels)
    ss_gp = sum(((pc1[(meta["group"] == g)].mean() - grand) ** 2) *
                (meta["group"] == g).sum() for g in gp_levels)
    ss_tot = float(np.var(pc1) * len(pc1))
    ss_res = max(0.0, ss_tot - ss_ds - ss_gp)

    parts = {
        "dataset": (ss_ds / ss_tot) * 100 if ss_tot else 0.0,
        "group":   (ss_gp / ss_tot) * 100 if ss_tot else 0.0,
        "resid":   (ss_res / ss_tot) * 100 if ss_tot else 0.0
    }

    if make_plots:
        def scatter_by(key, fname, title):
            plt.figure(figsize=(7, 6))
            for lvl in meta[key].unique():
                m = (meta[key] == lvl).values
                plt.scatter(Xp[m, 0], Xp[m, 1], s=50, alpha=0.85, label=str(lvl))
            plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
            plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
            plt.title(title)
            plt.legend(fontsize=8)
            plt.grid(True)
            plt.savefig(os.path.join(outdir, fname), bbox_inches="tight", dpi=300)
            plt.close()

        def scatter3d_by(key, fname, title):
            if Xp.shape[1] < 3:
                return
            fig = plt.figure(figsize=(8, 7))
            ax = fig.add_subplot(111, projection='3d')
            for lvl in meta[key].unique():
                m = (meta[key] == lvl).values
                ax.scatter(Xp[m, 0], Xp[m, 1], Xp[m, 2], s=40, alpha=0.85, label=str(lvl))
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
            ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)")
            ax.set_title(title)
            ax.legend(fontsize=8)
            ax.view_init(elev=20, azim=35)
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, fname), bbox_inches="tight", dpi=300)
            plt.close(fig)

        # 2D plots
        scatter_by("dataset", f"PCA_{tag}_by_dataset.png", f"PCA ({tag}) by dataset")
        scatter_by("group",   f"PCA_{tag}_by_group.png",   f"PCA ({tag}) by group")

        # 3D plots
        if make_plots_3d:
            scatter3d_by("dataset", f"PCA_{tag}_by_dataset_3D.png", f"PCA ({tag}) 3D by dataset")
            scatter3d_by("group",   f"PCA_{tag}_by_group_3D.png",   f"PCA ({tag}) 3D by group")

    return pca, Xp, parts


def _plotly_scatter3d_by(Xp, meta, pca, key, outdir, tag, title):
    """
    Write an interactive 3D PCA scatter HTML (Plotly) colored by 'key' (dataset/group).
    Avoid specifying colors explicitly so Plotly uses defaults.
    """
    if not _PLOTLY_OK or Xp.shape[1] < 3:
        return None

    traces = []
    for lvl in meta[key].unique():
        m = (meta[key] == lvl).values
        hover = [
            f"sample: {meta['column'].iloc[i]}<br>dataset: {meta['dataset'].iloc[i]}<br>group: {meta['group'].iloc[i]}"
            for i in np.where(m)[0]
        ]
        traces.append(go.Scatter3d(
            x=Xp[m, 0], y=Xp[m, 1], z=Xp[m, 2],
            mode="markers",
            marker=dict(size=4, opacity=0.9),
            name=str(lvl),
            text=hover,
            hoverinfo="text"
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
            zaxis_title=f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)",
        ),
        legend=dict(itemsizing="constant")
    )
    out_html = os.path.join(outdir, f"PCA_{tag}_by_{key}_3D_interactive.html")
    # include_plotlyjs=True -> self-contained HTML (offline-friendly)
    fig.write_html(out_html, include_plotlyjs=True, full_html=True)
    return out_html


def maybe_write_plotly_interactive(Xp, meta, pca, outdir, tag, enable):
    if not enable:
        return []
    if not _PLOTLY_OK:
        print("[WARN] Plotly not available. Skipping interactive 3D plots. Install with: pip install plotly")
        return []
    paths = []
    for key, ttl in [("dataset", f"PCA ({tag}) 3D by dataset — interactive"),
                     ("group",   f"PCA ({tag}) 3D by group — interactive")]:
        p = _plotly_scatter3d_by(Xp, meta, pca, key, outdir, tag, ttl)
        if p:
            paths.append(p)
    return paths


def remove_batch_effect(X, groups, datasets):
    """
    Remove dataset (batch) effects from X (samples x features) using:
        y ~ 1 + group + dataset
    Return corrected X (same shape). Group effects are kept, dataset effects removed.
    """
    # One-hot (drop_first) to avoid collinearity
    G = pd.get_dummies(pd.Categorical(groups), drop_first=True).to_numpy()
    B = pd.get_dummies(pd.Categorical(datasets), drop_first=True).to_numpy()

    # Design matrices
    n = X.shape[0]
    intercept = np.ones((n, 1), dtype=float)
    D_full  = np.concatenate([intercept, G, B], axis=1)
    D_batch = np.concatenate([np.zeros_like(intercept), np.zeros_like(G), B], axis=1) if B.size else np.zeros_like(D_full)

    # Solve per-feature and subtract batch part
    Xc = np.empty_like(X, dtype=float)
    for j in range(X.shape[1]):
        y = X[:, j]
        beta, *_ = np.linalg.lstsq(D_full, y, rcond=None)
        y_hat_batch = D_batch @ beta if B.size else 0.0
        Xc[:, j] = y - y_hat_batch
    return Xc


# --------------------------- Main --------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input)
    sample_cols = detect_sample_columns(df.columns, args.sample_suffix, args.ignore_regex)
    if not sample_cols:
        raise SystemExit("No sample columns detected. Adjust --sample_suffix / --ignore_regex.")
    print(f"[OK] Detected {len(sample_cols)} sample columns; {df.shape[0]} reactions.")

    # Build metadata table
    meta_rows = []
    for c in sample_cols:
        ds, gp = parse_meta_from_col(c, args.dataset_regex, args.group_regex)
        meta_rows.append({"column": c, "dataset": ds, "group": gp})
    meta = pd.DataFrame(meta_rows)
    meta_path = os.path.join(args.outdir, "sample_metadata.tsv")
    meta.to_csv(meta_path, sep="\t", index=False)
    print(f"[WRITE] Metadata -> {meta_path}")

    # Matrix samples x features
    X = df[sample_cols].T.values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Pre-correction diagnostics
    pca_pre, Xp_pre, parts_pre = pca_and_partition(
        X, meta, args.outdir, tag="pre",
        make_plots=(not args.no_plots),
        make_plots_3d=(not args.no_plots and not args.no_3d)
    )
    print(f"[PRE] PC1 partition: dataset={parts_pre['dataset']:.1f}% | group={parts_pre['group']:.1f}% | resid={parts_pre['resid']:.1f}%")
    interactive_pre = maybe_write_plotly_interactive(Xp_pre, meta, pca_pre, args.outdir, "pre", args.interactive_3d)
    for p in interactive_pre:
        print(f"[WRITE] Interactive 3D -> {p}")

    # Batch correction
    Xc = remove_batch_effect(X, meta["group"].values, meta["dataset"].values)

    # Post-correction diagnostics
    pca_post, Xp_post, parts_post = pca_and_partition(
        Xc, meta, args.outdir, tag="post",
        make_plots=(not args.no_plots),
        make_plots_3d=(not args.no_plots and not args.no_3d)
    )
    print(f"[POST] PC1 partition: dataset={parts_post['dataset']:.1f}% | group={parts_post['group']:.1f}% | resid={parts_post['resid']:.1f}%")
    interactive_post = maybe_write_plotly_interactive(Xp_post, meta, pca_post, args.outdir, "post", args.interactive_3d)
    for p in interactive_post:
        print(f"[WRITE] Interactive 3D -> {p}")

    # Write corrected CSV (replace *_Flux columns)
    df_out = df.copy()
    Xc_df = pd.DataFrame(Xc.T, columns=sample_cols)  # back to features x samples
    for c in sample_cols:
        df_out[c] = Xc_df[c].values
    df_out.to_csv(args.output_csv, index=False)
    print(f"[WRITE] Corrected CSV -> {args.output_csv}")

    # Write summary
    summary_path = os.path.join(args.outdir, "batch_correction_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Batch correction summary (dataset=batch; preserving groups)\n")
        f.write(f"Samples: {len(sample_cols)}; Reactions: {X.shape[1]}\n\n")
        f.write("Variance partition of PC1 (pre):\n")
        f.write(f"  Dataset (batch): {parts_pre['dataset']:.1f}%\n")
        f.write(f"  Group (biology): {parts_pre['group']:.1f}%\n")
        f.write(f"  Residual:        {parts_pre['resid']:.1f}%\n\n")
        f.write("Variance partition of PC1 (post):\n")
        f.write(f"  Dataset (batch): {parts_post['dataset']:.1f}%\n")
        f.write(f"  Group (biology): {parts_post['group']:.1f}%\n")
        f.write(f"  Residual:        {parts_post['resid']:.1f}%\n")
        if args.interactive_3d:
            if _PLOTLY_OK:
                f.write("\nInteractive 3D HTML files were saved alongside PNGs.\n")
            else:
                f.write("\n[WARN] Requested interactive 3D but Plotly is not available. Install with: pip install plotly\n")
        f.write("\nNote: 3D PCA PNGs are saved unless --no_3d is used.\n")
    print(f"[WRITE] Summary -> {summary_path}")

    print("\nNext step:")
    print("  python combined_analysis_interactive_generic.py \\")
    print(f"    --input {args.output_csv} \\")
    print("    --output_dir Combined_Analysis_Generic_BC")


if __name__ == "__main__":
    main()
