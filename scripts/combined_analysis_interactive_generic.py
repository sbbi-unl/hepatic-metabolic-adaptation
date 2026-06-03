
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  python combined_analysis_interactive_generic.py \
      --input combined_reaction_flux_comparison_extended.csv \
      --output_dir Combined_Analysis_Generic

  # If your sample columns end with something else:
  python combined_analysis_interactive_generic.py \
      --input my_combined.csv \
      --sample_suffix "_Flux" \
      --output_dir Analysis_OUT

  # If you prefer regex detection for sample columns:
  python combined_analysis_interactive_generic.py \
      --input my_combined.csv \
      --sample_regex ".*_Flux$" \
      --output_dir Analysis_OUT

  # Provide explicit regex for dataset IDs (defaults detect GSE/GSM):
  python combined_analysis_interactive_generic.py \
      --input my_combined.csv \
      --dataset_regex "(?:GSE|GSM)\\d+" \
      --output_dir Analysis_OUT
"""

import os
import re
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless save
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '-'

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# -------------------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generic cross-dataset PCA + clustering for combined flux tables."
    )
    p.add_argument("--input", "-i", required=True, help="Path to combined CSV.")
    p.add_argument("--output_dir", "-o", default="Combined_Analysis", help="Output directory.")
    # Column detection
    p.add_argument("--sample_suffix", default="_Flux",
                   help="Suffix that sample columns end with (ignored if --sample_regex present).")
    p.add_argument("--sample_regex", default=None,
                   help="Regex to *match* sample columns. If provided, overrides suffix logic.")
    p.add_argument("--ignore_regex", default="(MeanFlux|StdFlux|SEMFlux)",
                   help="Regex for columns to ignore (applies after sample detection).")
    # Metadata parsing
    p.add_argument("--dataset_regex", default=r"(?:GSE|GSM)\d+",
                   help="Regex to detect dataset identifiers inside column names.")
    p.add_argument("--group_regex", default=None,
                   help="Optional regex to detect group label. If omitted, uses prefix before first underscore.")
    # PCA options
    p.add_argument("--max_components", type=int, default=10,
                   help="Max PCA components (bounded by n_samples-1 internally).")
    p.add_argument("--top_loading_reactions", type=int, default=30,
                   help="Top reactions to show in loadings heatmap.")
    # Interactivity toggles
    p.add_argument("--no_interactive", action="store_true", help="Skip Plotly HTML outputs.")
    return p.parse_args()


def nice_print_header(title):
    line = "=" * 80
    print("\n" + line)
    print(title)
    print(line)


def detect_sample_columns(df, sample_regex, sample_suffix, ignore_regex):
    cols = []
    patt = re.compile(sample_regex) if sample_regex else None
    ignore = re.compile(ignore_regex) if ignore_regex else None

    for c in df.columns:
        is_sample = False
        if patt:
            if patt.search(str(c)):
                is_sample = True
        else:
            if str(c).endswith(sample_suffix):
                is_sample = True

        if is_sample and ignore and ignore.search(str(c)):
            is_sample = False

        if is_sample:
            cols.append(c)

    return cols


def parse_column_metadata(columns, dataset_regex, group_regex):
    """
    Extract metadata (group, dataset, full_name) from column names.

    Heuristics:
    - group: if group_regex given -> first match; else string before first underscore
    - dataset: first token matching dataset_regex; else 'Unknown'
    - full_name: column name with trailing sample suffix removed when possible
    """
    meta_rows = []
    ds_re = re.compile(dataset_regex) if dataset_regex else None
    grp_re = re.compile(group_regex) if group_regex else None

    for col in columns:
        col_str = str(col)

        # full_name: strip a trailing "_Flux" or "_<suffix>" if present
        full_name = re.sub(r"(_Flux|_flux)$", "", col_str)

        # dataset
        dataset = None
        if ds_re:
            m = ds_re.search(col_str)
            if m:
                dataset = m.group(0)
        if not dataset:
            dataset = "Unknown"

        # group
        group = None
        if grp_re:
            g = grp_re.search(col_str)
            if g:
                group = g.group(0)
        if not group:
            # default heuristic = prefix before first underscore
            group = col_str.split("_", 1)[0]

        meta_rows.append({
            "column": col_str,
            "group": group,
            "dataset": dataset,
            "full_name": full_name
        })

    meta = pd.DataFrame(meta_rows)
    for k in ["column", "group", "dataset", "full_name"]:
        meta[k] = meta[k].astype(str)
    return meta


def _to_mpl_color(c):
    """Convert Plotly 'rgb(r,g,b)' strings or RGB triplets into Matplotlib-friendly hex."""
    if isinstance(c, str):
        s = c.strip()
        m = re.match(r'rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)$', s)
        if m:
            r, g, b = map(int, m.groups())
            return mcolors.to_hex((r/255.0, g/255.0, b/255.0))
        # already a name/hex
        return s
    if isinstance(c, (tuple, list)) and len(c) >= 3:
        r, g, b = c[:3]
        if max(r, g, b) > 1.0:
            r, g, b = r/255.0, g/255.0, b/255.0
        return mcolors.to_hex((r, g, b))
    return c


def build_color_map(keys, palette):
    # repeat palette if needed and convert to mpl-friendly
    colors = {}
    for i, k in enumerate(keys):
        raw = palette[i % len(palette)]
        colors[k] = _to_mpl_color(raw)
    return colors


def mpl_set_labels(ax, x, y, title):
    ax.set_xlabel(x, fontweight="bold")
    ax.set_ylabel(y, fontweight="bold")
    ax.set_title(title, fontweight="bold", loc="left")


def savefig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------------------------
# Core analyses
# -------------------------------------------------------------------------------------

def load_and_prepare(input_csv, sample_regex, sample_suffix, ignore_regex,
                     dataset_regex, group_regex):
    nice_print_header("LOADING DATA")
    df = pd.read_csv(input_csv)
    #print(f"✓ Loaded: {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"[OK] Loaded: {df.shape[0]} rows x {df.shape[1]} cols")


    # Detect sample columns
    sample_cols = detect_sample_columns(df, sample_regex, sample_suffix, ignore_regex)
    if not sample_cols:
        raise ValueError(
            "No sample columns were detected. Consider adjusting --sample_suffix or --sample_regex."
        )
    print(f"[OK] Detected {len(sample_cols)} sample columns")

    # Parse metadata
    meta = parse_column_metadata(sample_cols, dataset_regex, group_regex)
    print("Samples per dataset:")
    for ds in meta["dataset"].unique():
        n = (meta["dataset"] == ds).sum()
        groups = sorted(meta.loc[meta["dataset"] == ds, "group"].unique().tolist())
        print(f"  {ds}: n={n} ({', '.join(groups)})")

    print("\nSamples per group:")
    for g in meta["group"].unique():
        n = (meta["group"] == g).sum()
        ds_ct = meta.loc[meta["group"] == g, "dataset"].nunique()
        print(f"  {g}: n={n} from {ds_ct} dataset(s)")

    # Construct matrix: samples × features
    X = df[sample_cols].T.values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return df, meta, X


def cross_dataset_pca(df, meta, X, max_components, top_loading_reactions, outdir):
    nice_print_header("CROSS-DATASET PCA")

    os.makedirs(outdir, exist_ok=True)

    # Standardize
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    ncomp = max(2, min(max_components, Xs.shape[0] - 1))
    pca = PCA(n_components=ncomp)
    Xp = pca.fit_transform(Xs)

    # append PCs into meta
    for i in range(min(5, ncomp)):
        meta[f"PC{i+1}"] = Xp[:, i]

    unique_datasets = meta["dataset"].unique().tolist()
    unique_groups = meta["group"].unique().tolist()

    # Color maps (keep aesthetic similar)
    dataset_colors = build_color_map(unique_datasets, px.colors.qualitative.Set2)
    group_colors = build_color_map(unique_groups, px.colors.qualitative.Bold)

    # --- Composite figure with 10 panels (A..J) ---
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.35)

    # A: By dataset
    ax1 = fig.add_subplot(gs[0, 0])
    for ds in unique_datasets:
        mask = (meta["dataset"] == ds).values
        ax1.scatter(Xp[mask, 0], Xp[mask, 1],
                    s=80, alpha=0.7,
                    edgecolors='black', linewidth=0.5,
                    c=dataset_colors[ds], label=ds)
    mpl_set_labels(ax1,
                   f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                   f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
                   "A. Samples by Dataset (Batch Effect Check)")
    ax1.legend(fontsize=7, loc='best')

    # B: By group
    ax2 = fig.add_subplot(gs[0, 1])
    for g in unique_groups:
        mask = (meta["group"] == g).values
        ax2.scatter(Xp[mask, 0], Xp[mask, 1],
                    s=80, alpha=0.7,
                    edgecolors='black', linewidth=0.5,
                    c=group_colors[g], label=g)
    mpl_set_labels(ax2,
                   f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                   f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
                   "B. Samples by Group (Biological Signal)")
    ax2.legend(fontsize=9, loc='best')

    # C: Combined (shape by dataset, color by group)
    ax3 = fig.add_subplot(gs[0, 2])
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X']
    for i, ds in enumerate(unique_datasets):
        for g in unique_groups:
            mask = ((meta["dataset"] == ds) & (meta["group"] == g)).values
            if mask.sum() > 0:
                ax3.scatter(Xp[mask, 0], Xp[mask, 1],
                            marker=markers[i % len(markers)],
                            s=80, alpha=0.7,
                            edgecolors='black', linewidth=0.5,
                            c=group_colors[g])
    legend_patches = [mpatches.Patch(color=group_colors[g], label=g) for g in unique_groups]
    ax3.legend(handles=legend_patches, fontsize=9, loc='best', title='Group')
    mpl_set_labels(ax3,
                   f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                   f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
                   "C. Combined (Shape=Dataset, Color=Group)")

    # D: Scree
    ax4 = fig.add_subplot(gs[0, 3])
    n_show = min(15, ncomp)
    ax4.bar(range(1, n_show+1),
            pca.explained_variance_ratio_[:n_show]*100,
            edgecolor='black')
    ax4.plot(range(1, n_show+1),
             np.cumsum(pca.explained_variance_ratio_[:n_show])*100,
             'o-', linewidth=2)
    mpl_set_labels(ax4, 'Principal Component', 'Variance Explained (%)', 'D. Scree Plot')
    ax4.set_xticks(range(1, n_show+1))

    # E: PC1 vs PC3
    if ncomp >= 3:
        ax5 = fig.add_subplot(gs[1, 0])
        for g in unique_groups:
            mask = (meta["group"] == g).values
            ax5.scatter(Xp[mask, 0], Xp[mask, 2],
                        s=80, alpha=0.7,
                        edgecolors='black', linewidth=0.5,
                        c=group_colors[g], label=g)
        mpl_set_labels(ax5,
                       f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                       f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)',
                       'E. PC1 vs PC3')
        ax5.legend(fontsize=8, loc='best')

    # F: PC2 vs PC3
    if ncomp >= 3:
        ax6 = fig.add_subplot(gs[1, 1])
        for g in unique_groups:
            mask = (meta["group"] == g).values
            ax6.scatter(Xp[mask, 1], Xp[mask, 2],
                        s=80, alpha=0.7,
                        edgecolors='black', linewidth=0.5,
                        c=group_colors[g], label=g)
        mpl_set_labels(ax6,
                       f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
                       f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)',
                       'F. PC2 vs PC3')
        ax6.legend(fontsize=8, loc='best')

    # G: Boxplot PC1 by dataset
    ax7 = fig.add_subplot(gs[1, 2])
    pc1_data = []
    labels = []
    for ds in unique_datasets:
        mask = (meta["dataset"] == ds).values
        pc1_data.append(Xp[mask, 0])
        labels.append(ds.replace('GSE', ''))
    bp = ax7.boxplot(pc1_data, labels=labels, patch_artist=True)
    for patch, ds in zip(bp['boxes'], unique_datasets):
        patch.set_facecolor(dataset_colors[ds])
        patch.set_alpha(0.7)
    mpl_set_labels(ax7, 'Dataset', 'PC1 Score', 'G. PC1 Distribution by Dataset')
    ax7.tick_params(axis='x', rotation=45)

    # H: Boxplot PC1 by group
    ax8 = fig.add_subplot(gs[1, 3])
    pc1_group_data, group_labels = [], []
    for g in unique_groups:
        mask = (meta["group"] == g).values
        pc1_group_data.append(Xp[mask, 0])
        group_labels.append(g)
    bp2 = ax8.boxplot(pc1_group_data, labels=group_labels, patch_artist=True)
    for patch, g in zip(bp2['boxes'], unique_groups):
        patch.set_facecolor(group_colors[g])
        patch.set_alpha(0.7)
    mpl_set_labels(ax8, 'Group', 'PC1 Score', 'H. PC1 Distribution by Group')

    # I: Loadings heatmap (top reactions across PC1..3 by |loading| sum)
    ax9 = fig.add_subplot(gs[2, :2])
    comps = pca.components_[:min(3, ncomp), :]  # first 3 PCs or less
    top_indices = np.argsort(np.abs(comps).sum(axis=0))[-top_loading_reactions:]
    if "ReactionName" in df.columns:
        ytick = [str(df.iloc[i]["ReactionName"])[:40] if pd.notna(df.iloc[i]["ReactionName"]) else
                 str(df.iloc[i].get("ReactionID", f"Rxn_{i}"))[:40] for i in top_indices]
    elif "ReactionID" in df.columns:
        ytick = [str(df.iloc[i]["ReactionID"])[:40] for i in top_indices]
    else:
        ytick = [f"Rxn_{i}" for i in top_indices]
    im = ax9.imshow(comps[:, top_indices].T, aspect="auto", cmap="RdBu_r",
                    vmin=-np.max(np.abs(comps)), vmax=np.max(np.abs(comps)))
    ax9.set_yticks(range(len(ytick)))
    ax9.set_yticklabels(ytick, fontsize=8)
    ax9.set_xticks(range(comps.shape[0]))
    ax9.set_xticklabels([f"PC{k+1}" for k in range(comps.shape[0])])
    ax9.set_title("I. Top Reactions Driving PCA Separation", fontweight="bold", loc="left")
    cbar = fig.colorbar(im, ax=ax9)
    cbar.set_label("Loading")

    # J: Variance partition (PC1)
    ax10 = fig.add_subplot(gs[2, 2:])
    grand_mean = Xp[:, 0].mean()
    ss_dataset = sum(((Xp[(meta["dataset"] == ds).values, 0].mean() - grand_mean) ** 2) *
                     (meta["dataset"] == ds).sum()
                     for ds in unique_datasets)
    ss_group = sum(((Xp[(meta["group"] == g).values, 0].mean() - grand_mean) ** 2) *
                   (meta["group"] == g).sum()
                   for g in unique_groups)
    ss_total = float(np.var(Xp[:, 0]) * len(Xp))
    ss_resid = max(0.0, ss_total - ss_dataset - ss_group)
    parts = {
        "Dataset\n(Batch)": (ss_dataset / ss_total) * 100 if ss_total else 0.0,
        "Group\n(Biology)": (ss_group / ss_total) * 100 if ss_total else 0.0,
        "Residual": (ss_resid / ss_total) * 100 if ss_total else 0.0
    }
    bars = ax10.bar(list(parts.keys()), list(parts.values()), edgecolor="black", linewidth=2)
    ax10.set_ylim(0, 100)
    mpl_set_labels(ax10, "", "% Variance Explained (PC1)", "J. Variance Partitioning: Batch vs Biology")
    for b, (k, v) in zip(bars, parts.items()):
        b.set_facecolor({"Dataset\n(Batch)": "#e74c3c", "Group\n(Biology)": "#3498db"}.get(k, "#95a5a6"))
        ax10.text(b.get_x() + b.get_width()/2.0, b.get_height() + 2, f"{v:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)

    plt.tight_layout()
    savefig(fig, os.path.join(outdir, "Figure_CrossDataset_PCA_Comprehensive.png"))

    # Export PCA coordinates
    export_cols = ["column", "group", "dataset", "full_name"] + [f"PC{i+1}" for i in range(min(5, ncomp))]
    meta[export_cols].to_csv(os.path.join(outdir, "pca_coordinates.csv"), index=False)

    # Diagnostics to stdout
    print("\nVARIANCE PARTITION (PC1):")
    for k, v in parts.items():
        print(f"  {k.replace(chr(10),' ')}: {v:.1f}%")
    if parts["Group\n(Biology)"] > parts["Dataset\n(Batch)"]:
        print("[OK] Biological signal dominates — good!")
    else:
        print("[Attention] Batch effects dominate — consider batch correction.")

    return pca, Xp, meta, dataset_colors, group_colors


def save_individual_panels(Xp, pca, meta, dataset_colors, group_colors, outdir):
    os.makedirs(outdir, exist_ok=True)
    unique_datasets = meta["dataset"].unique().tolist()
    unique_groups = meta["group"].unique().tolist()
    ncomp = Xp.shape[1]

    # A
    fig, ax = plt.subplots(figsize=(8, 6))
    for ds in unique_datasets:
        m = (meta["dataset"] == ds).values
        ax.scatter(Xp[m, 0], Xp[m, 1], s=100, alpha=0.7,
                   edgecolors='black', linewidth=0.5,
                   c=dataset_colors[ds], label=ds)
    mpl_set_labels(ax, f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                   f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
                   "Samples by Dataset (Batch Effect Check)")
    ax.legend(fontsize=9, loc='best')
    plt.tight_layout()
    savefig(fig, os.path.join(outdir, "Panel_A_PCA_by_Dataset.png"))

    # B
    fig, ax = plt.subplots(figsize=(8, 6))
    for g in unique_groups:
        m = (meta["group"] == g).values
        ax.scatter(Xp[m, 0], Xp[m, 1], s=100, alpha=0.7,
                   edgecolors='black', linewidth=0.5,
                   c=group_colors[g], label=g)
    mpl_set_labels(ax, f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                   f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
                   "Samples by Group (Biological Signal)")
    ax.legend(fontsize=10, loc='best')
    plt.tight_layout()
    savefig(fig, os.path.join(outdir, "Panel_B_PCA_by_Group.png"))

    # C
    fig, ax = plt.subplots(figsize=(8, 6))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X']
    for i, ds in enumerate(unique_datasets):
        for g in unique_groups:
            m = ((meta["dataset"] == ds) & (meta["group"] == g)).values
            if m.sum() > 0:
                ax.scatter(Xp[m, 0], Xp[m, 1],
                           marker=markers[i % len(markers)],
                           s=100, alpha=0.7,
                           edgecolors='black', linewidth=0.5,
                           c=group_colors[g])
    legend_patches = [mpatches.Patch(color=group_colors[g], label=g) for g in unique_groups]
    ax.legend(handles=legend_patches, fontsize=10, loc='best', title='Group')
    mpl_set_labels(ax, f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                   f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
                   "Combined View (Shape=Dataset, Color=Group)")
    plt.tight_layout()
    savefig(fig, os.path.join(outdir, "Panel_C_Combined_View.png"))

    # D
    fig, ax = plt.subplots(figsize=(8, 6))
    n_show = min(15, ncomp)
    ax.bar(range(1, n_show+1),
           pca.explained_variance_ratio_[:n_show]*100, edgecolor='black', linewidth=1.5)
    ax.plot(range(1, n_show+1),
            np.cumsum(pca.explained_variance_ratio_[:n_show])*100, 'o-', linewidth=2)
    mpl_set_labels(ax, "Principal Component", "Variance Explained (%)", "Scree Plot")
    ax.set_xticks(range(1, n_show+1))
    plt.tight_layout()
    savefig(fig, os.path.join(outdir, "Panel_D_Scree_Plot.png"))

    # E
    if ncomp >= 3:
        fig, ax = plt.subplots(figsize=(8, 6))
        for g in unique_groups:
            m = (meta["group"] == g).values
            ax.scatter(Xp[m, 0], Xp[m, 2], s=100, alpha=0.7,
                       edgecolors='black', linewidth=0.5,
                       c=group_colors[g], label=g)
        mpl_set_labels(ax, f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                       f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)',
                       "PC1 vs PC3")
        ax.legend(fontsize=9, loc='best')
        plt.tight_layout()
        savefig(fig, os.path.join(outdir, "Panel_E_PC1_vs_PC3.png"))

    # F
    if ncomp >= 3:
        fig, ax = plt.subplots(figsize=(8, 6))
        for g in unique_groups:
            m = (meta["group"] == g).values
            ax.scatter(Xp[m, 1], Xp[m, 2], s=100, alpha=0.7,
                       edgecolors='black', linewidth=0.5,
                       c=group_colors[g], label=g)
        mpl_set_labels(ax, f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
                       f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)',
                       "PC2 vs PC3")
        ax.legend(fontsize=9, loc='best')
        plt.tight_layout()
        savefig(fig, os.path.join(outdir, "Panel_F_PC2_vs_PC3.png"))

    # G
    fig, ax = plt.subplots(figsize=(8, 6))
    pc1 = []
    labels = []
    for ds in unique_datasets:
        m = (meta["dataset"] == ds).values
        pc1.append(Xp[m, 0])
        labels.append(ds.replace("GSE", ""))
    bp = ax.boxplot(pc1, labels=labels, patch_artist=True)
    for patch, ds in zip(bp['boxes'], unique_datasets):
        patch.set_facecolor(dataset_colors[ds])
        patch.set_alpha(0.7)
    mpl_set_labels(ax, "Dataset", "PC1 Score", "PC1 Distribution by Dataset")
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    savefig(fig, os.path.join(outdir, "Panel_G_PC1_by_Dataset.png"))

    # H
    fig, ax = plt.subplots(figsize=(8, 6))
    pc1g, glabels = [], []
    for g in unique_groups:
        m = (meta["group"] == g).values
        pc1g.append(Xp[m, 0])
        glabels.append(g)
    bp2 = ax.boxplot(pc1g, labels=glabels, patch_artist=True)
    for patch, g in zip(bp2['boxes'], unique_groups):
        patch.set_facecolor(group_colors[g])
        patch.set_alpha(0.7)
    mpl_set_labels(ax, "Group", "PC1 Score", "PC1 Distribution by Group")
    plt.tight_layout()
    savefig(fig, os.path.join(outdir, "Panel_H_PC1_by_Group.png"))

    # I and J are covered in the composite figure.


def interactive_3d_plots(Xp, meta, pca, outdir):
    if Xp.shape[1] < 3:
        print("Skipping interactive 3D plots (need at least 3 PCs).")
        return

    os.makedirs(outdir, exist_ok=True)

    df_plot = pd.DataFrame({
        "PC1": Xp[:, 0],
        "PC2": Xp[:, 1],
        "PC3": Xp[:, 2],
        "Group": meta["group"].values,
        "Dataset": meta["dataset"].values,
        "Sample": meta["full_name"].values
    })

    var_pc1 = pca.explained_variance_ratio_[0] * 100
    var_pc2 = pca.explained_variance_ratio_[1] * 100
    var_pc3 = pca.explained_variance_ratio_[2] * 100
    cum = (pca.explained_variance_ratio_[:3].sum()) * 100

    # 1) Colored by group
    fig1 = go.Figure()
    groups = df_plot["Group"].unique().tolist()
    palette = px.colors.qualitative.Bold
    gcolors = build_color_map(groups, palette)  # produces hex for mpl, fine in plotly too
    for g in groups:
        d = df_plot[df_plot["Group"] == g]
        fig1.add_trace(go.Scatter3d(
            x=d["PC1"], y=d["PC2"], z=d["PC3"],
            mode="markers", name=f"{g} (n={len(d)})",
            marker=dict(size=8, color=gcolors[g], opacity=0.85,
                        line=dict(color="black", width=0.5)),
            text=d["Sample"],
            hovertemplate="<b>%{text}</b><br>PC1=%{x:.2f}<br>PC2=%{y:.2f}<br>PC3=%{z:.2f}<extra></extra>"
        ))
    fig1.update_layout(
        title=f"Interactive 3D PCA by Group<br><sub>Cumulative variance: {cum:.1f}%</sub>",
        scene=dict(
            xaxis_title=f"PC1 ({var_pc1:.1f}%)",
            yaxis_title=f"PC2 ({var_pc2:.1f}%)",
            zaxis_title=f"PC3 ({var_pc3:.1f}%)"
        ),
        width=1200, height=800, hovermode="closest"
    )
    fig1.write_html(os.path.join(outdir, "Interactive_3D_PCA_by_Group.html"))

    # 2) Colored by dataset
    fig2 = go.Figure()
    datasets = df_plot["Dataset"].unique().tolist()
    dcolors = build_color_map(datasets, px.colors.qualitative.Set2)
    for ds in datasets:
        d = df_plot[df_plot["Dataset"] == ds]
        fig2.add_trace(go.Scatter3d(
            x=d["PC1"], y=d["PC2"], z=d["PC3"],
            mode="markers", name=f"{ds} (n={len(d)})",
            marker=dict(size=8, color=dcolors[ds], opacity=0.85,
                        line=dict(color="black", width=0.5)),
            text=d["Sample"],
            hovertemplate="<b>%{text}</b><br>PC1=%{x:.2f}<br>PC2=%{y:.2f}<br>PC3=%{z:.2f}<extra></extra>"
        ))
    fig2.update_layout(
        title="Interactive 3D PCA by Dataset (Batch Check)",
        scene=dict(
            xaxis_title=f"PC1 ({var_pc1:.1f}%)",
            yaxis_title=f"PC2 ({var_pc2:.1f}%)",
            zaxis_title=f"PC3 ({var_pc3:.1f}%)"
        ),
        width=1200, height=800, hovermode="closest"
    )
    fig2.write_html(os.path.join(outdir, "Interactive_3D_PCA_by_Dataset.html"))

    # 3) Side-by-side
    fig3 = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
        subplot_titles=("Colored by Group", "Colored by Dataset"),
        horizontal_spacing=0.05
    )
    for g in groups:
        d = df_plot[df_plot["Group"] == g]
        fig3.add_trace(
            go.Scatter3d(
                x=d["PC1"], y=d["PC2"], z=d["PC3"], mode="markers", name=g,
                marker=dict(size=6, color=gcolors[g], opacity=0.8, line=dict(color="black", width=0.3)),
                text=d["Sample"],
                hovertemplate="<b>%{text}</b><br>PC1=%{x:.2f}<br>PC2=%{y:.2f}<br>PC3=%{z:.2f}<extra></extra>"
            ), row=1, col=1
        )
    for ds in datasets:
        d = df_plot[df_plot["Dataset"] == ds]
        fig3.add_trace(
            go.Scatter3d(
                x=d["PC1"], y=d["PC2"], z=d["PC3"], mode="markers", name=ds,
                marker=dict(size=6, color=dcolors[ds], opacity=0.8, line=dict(color="black", width=0.3)),
                text=d["Sample"],
                hovertemplate="<b>%{text}</b><br>PC1=%{x:.2f}<br>PC2=%{y:.2f}<br>PC3=%{z:.2f}<extra></extra>"
            ), row=1, col=2
        )
    for col in [1, 2]:
        fig3.update_scenes(
            xaxis_title=f"PC1 ({var_pc1:.1f}%)",
            yaxis_title=f"PC2 ({var_pc2:.1f}%)",
            zaxis_title=f"PC3 ({var_pc3:.1f}%)",
            row=1, col=col
        )
    fig3.update_layout(width=1800, height=800, hovermode="closest",
                       title="Interactive 3D PCA: Biology vs Batch")
    fig3.write_html(os.path.join(outdir, "Interactive_3D_PCA_Combined.html"))


def hierarchical_clustering(df, meta, X, outdir):
    nice_print_header("HIERARCHICAL CLUSTERING")

    os.makedirs(outdir, exist_ok=True)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    link = linkage(Xs, method="ward", metric="euclidean")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))
    labels = [fn.replace("_", " ") for fn in meta["full_name"].tolist()]
    dend = dendrogram(link, labels=labels, ax=ax1, leaf_font_size=7, leaf_rotation=90)
    ax1.set_ylabel("Distance", fontweight="bold")
    ax1.set_title("A. Hierarchical Clustering of All Samples", fontweight="bold", loc="left")

    # Quick 4-cluster cut visual (can be adjusted)
    k = min(4, Xs.shape[0])
    clusters = fcluster(link, t=k, criterion="maxclust")
    ax2.scatter(range(len(clusters)), clusters, s=40, edgecolors="black", linewidth=0.5)
    ax2.set_yticks(sorted(np.unique(clusters)))
    mpl_set_labels(ax2, "Sample Index", "Cluster ID", "B. Cluster Assignments (k≈4)")

    plt.tight_layout()
    savefig(fig, os.path.join(outdir, "Figure_Hierarchical_Clustering.png"))


# -------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df, meta, X = load_and_prepare(
        args.input,
        args.sample_regex,
        args.sample_suffix,
        args.ignore_regex,
        args.dataset_regex,
        args.group_regex
    )

    pca, Xp, meta, dcolors, gcolors = cross_dataset_pca(
        df=df, meta=meta, X=X,
        max_components=args.max_components,
        top_loading_reactions=args.top_loading_reactions,
        outdir=args.output_dir
    )

    # Save individual panels
    save_individual_panels(
        Xp=Xp, pca=pca, meta=meta,
        dataset_colors=dcolors, group_colors=gcolors,
        outdir=os.path.join(args.output_dir, "Individual_PCA_Panels")
    )

    # Interactive
    if not args.no_interactive:
        interactive_3d_plots(Xp, meta, pca, outdir=args.output_dir)

    # Clustering
    hierarchical_clustering(df, meta, X, outdir=args.output_dir)

    print("\nDone. Outputs written to:", os.path.abspath(args.output_dir))
    print("  - Figure_CrossDataset_PCA_Comprehensive.png")
    print("  - Individual_PCA_Panels/*.png")
    if not args.no_interactive:
        print("  - Interactive_3D_PCA_by_Group.html")
        print("  - Interactive_3D_PCA_by_Dataset.html")
        print("  - Interactive_3D_PCA_Combined.html")
    print("  - Figure_Hierarchical_Clustering.png")
    print("  - pca_coordinates.csv")


if __name__ == "__main__":
    main()
