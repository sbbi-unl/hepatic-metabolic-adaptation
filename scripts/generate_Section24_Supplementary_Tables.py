"""
generate_Section24_Supplementary_Tables.py
==========================================
Generates: Section24_Supplementary_Tables.xlsx

Input files (loaded dynamically from their respective subfolders):
    flux_comparison.csv                   — per-cell-type flux values (Chow vs WD)
    phase1_bulk_cellular_overlap.csv      — bulk vs cell overlap metrics
    statistical_tests.csv                 — all per-cell significant reactions
    phase2_contribution_analysis.csv      — cell-type contribution scores

Sheets produced:
    SJ1 — Composite cell flux vs bulk SCD: all 3726 reactions with
           composite (weighted) Chow flux and per-cell Chow fluxes
    SJ2 — Composite vs bulk Δflux correlation: WD-Chow changes
    SJ3 — Bulk-cellular overlap statistics per cell type
    SJ4 — Subsystem-level bulk vs cellular correlation summary
"""

import os, sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# ── Configuration ──────────────────────────────────────────────────────────────
# Base directory for inputs (assuming script runs from project root)
BASE_DIR    = Path("Processing_outputs")

# Output directory
OUTPUT_DIR  = BASE_DIR / "Supplementary_Tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "Section24_Supplementary_Tables.xlsx"

# ── Helpers ────────────────────────────────────────────────────────────────────
def load(rel_path):
    """Loads a CSV relative to the BASE_DIR."""
    path = BASE_DIR / rel_path
    if not path.exists():
        sys.exit(f"ERROR: Required file not found: {path}\nEnsure you are running this from the main project folder.")
    return pd.read_csv(path)

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading data files for Section 2.4...")

# Load from specific nested subdirectories
flux_cmp  = load("Step_3_RQ3/statistics/flux_comparison.csv")
stat_test = load("Step_3_RQ3/statistics/statistical_tests.csv")
overlap   = load("Step_3_RQ3/Step_3b_results_rq1_rq3_integration/tables/phase1_bulk_cellular_overlap.csv")
contrib   = load("Step_3_RQ3/Step_3b_results_rq1_rq3_integration/tables/phase2_contribution_analysis.csv")

# ── Identify Chow and WD columns ──────────────────────────────────────────────
chow_cols = [c for c in flux_cmp.columns if c.endswith("_Chow")]
wd_cols   = [c for c in flux_cmp.columns if c.endswith("_WesternDiet")]
cell_types_in_flux = [c.replace("_Chow","") for c in chow_cols]
print(f"  Cell types in flux_comparison: {len(cell_types_in_flux)}")

# ── Get cell abundances for weighting ─────────────────────────────────────────
abundance_map = dict(zip(contrib["cell_type"], contrib["cell_abundance"]))

# ── SJ1: Composite Chow flux and per-cell Chow fluxes ────────────────────────
print("Building SJ1: Composite Chow flux...")
sj1 = flux_cmp[["reaction_id","reaction_name","subsystem"] + chow_cols].copy()

# Compute abundance-weighted composite flux (only for cell types with abundance data)
weights = []
valid_cols = []
for ct, col in zip(cell_types_in_flux, chow_cols):
    ab = abundance_map.get(ct, None)
    if ab is not None and ab > 0:
        weights.append(ab)
        valid_cols.append(col)

if valid_cols:
    weight_arr = np.array(weights)
    weight_arr = weight_arr / weight_arr.sum()   # normalise to 1
    composite = (flux_cmp[valid_cols].values * weight_arr).sum(axis=1)
    sj1.insert(3, "Composite_Chow_Flux_Weighted", composite)

# Rename columns for clarity
rename_chow = {c: c.replace("_Chow","_ChowFlux") for c in chow_cols}
sj1 = sj1.rename(columns=rename_chow)
print(f"  SJ1: {len(sj1)} reactions × {len(sj1.columns)} columns")

# ── SJ2: Composite vs bulk Δflux ─────────────────────────────────────────────
print("Building SJ2: Δflux comparison...")
sj2 = flux_cmp[["reaction_id","reaction_name","subsystem"] + chow_cols + wd_cols].copy()

# Composite Δflux (WD − Chow) for each cell type
for ct, c_col, w_col in zip(cell_types_in_flux, chow_cols, wd_cols):
    delta_col = ct + "_DeltaFlux"
    sj2[delta_col] = sj2[w_col] - sj2[c_col]

# Drop raw Chow/WD columns, keep only deltas
delta_cols = [ct + "_DeltaFlux" for ct in cell_types_in_flux]
sj2 = sj2[["reaction_id","reaction_name","subsystem"] + delta_cols].copy()

# Weighted composite Δflux
if valid_cols:
    delta_valid = [ct + "_DeltaFlux" for ct in
                   [c.replace("_Chow","") for c in valid_cols]]
    comp_delta = (sj2[delta_valid].values * weight_arr).sum(axis=1)
    sj2.insert(3, "Composite_DeltaFlux_Weighted", comp_delta)

print(f"  SJ2: {len(sj2)} reactions × {len(sj2.columns)} columns")

# ── SJ3: Bulk-cellular overlap ────────────────────────────────────────────────
print("Building SJ3: Bulk-cellular overlap...")
sj3 = overlap.copy()
sj3.columns = [
    "Cell_Type", "N_Bulk_Significant", "N_Cell_Significant",
    "N_Overlap", "N_Bulk_Only", "N_Cell_Only",
    "Overlap_Pct", "Sensitivity", "Precision", "F1_Score", "Cell_Abundance"
]
# Add contribution percent from phase2
cp_map = dict(zip(contrib["cell_type"], contrib["contribution_percent"]))
sj3["Contribution_Percent"] = sj3["Cell_Type"].map(cp_map)
sj3 = sj3.sort_values("F1_Score", ascending=False)
print(f"  SJ3: {len(sj3)} cell types")

# ── SJ4: Subsystem-level correlation summary ──────────────────────────────────
print("Building SJ4: Subsystem correlation summary...")
# For each subsystem, compute correlation between composite cell Δflux and
# the significant cell reactions, summarised by subsystem
subsystem_data = []
subsystems = flux_cmp["subsystem"].unique()

if "Composite_DeltaFlux_Weighted" in sj2.columns:
    for sub in sorted(subsystems):
        sub_mask = sj2["subsystem"] == sub
        sub_df = sj2[sub_mask]
        if len(sub_df) < 3:
            continue
        # Compute average Δflux per reaction across cell types
        delta_arr = sub_df["Composite_DeltaFlux_Weighted"].values
        n_nonzero = (delta_arr != 0).sum()
        n_positive = (delta_arr > 0).sum()
        n_negative = (delta_arr < 0).sum()
        mean_delta = delta_arr.mean()
        subsystem_data.append({
            "Subsystem":           sub,
            "N_Reactions":         len(sub_df),
            "N_NonZero_Delta":     n_nonzero,
            "N_Up_Delta":          n_positive,
            "N_Down_Delta":        n_negative,
            "Mean_Composite_Delta": round(mean_delta, 6),
            "Dominant_Direction":  "Up" if n_positive > n_negative else
                                   "Down" if n_negative > n_positive else "Mixed",
        })
    sj4 = pd.DataFrame(subsystem_data).sort_values("N_NonZero_Delta", ascending=False)
else:
    sj4 = pd.DataFrame({"Note": ["Composite flux column not computed — check cell abundance data"]})

print(f"  SJ4: {len(sj4)} subsystems")

# ── Write Excel ────────────────────────────────────────────────────────────────
print(f"\nWriting {OUTPUT_FILE.name} ...")

SHEET_CONFIGS = [
    (sj1, "SJ1_Composite_Chow_Flux",    "#1F497D",
     "Abundance-weighted composite Chow flux and per-cell-type Chow flux values for all 3726 reactions"),
    (sj2, "SJ2_DeltaFlux_Comparison",   "#2F4F8F",
     "Composite and per-cell-type WD-minus-Chow delta flux for all 3726 reactions"),
    (sj3, "SJ3_Bulk_Cellular_Overlap",  "#4B0082",
     "Overlap between bulk RQ1 significant reactions and single-cell RQ3 significant reactions per cell type"),
    (sj4, "SJ4_Subsystem_Summary",      "#006400",
     "Subsystem-level summary of composite delta flux direction and magnitude"),
]

with pd.ExcelWriter(OUTPUT_FILE, engine="xlsxwriter") as writer:
    wb = writer.book

    def styled_sheet(df, sheet_name, header_color, description):
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        ws = writer.sheets[sheet_name]
        hf = wb.add_format({
            "bold": True, "bg_color": header_color, "font_color": "white",
            "border": 1, "text_wrap": True
        })
        for i, col in enumerate(df.columns):
            ws.write(0, i, col, hf)
            max_w = max(len(str(col)) + 2,
                        df[col].astype(str).str.len().max() + 2 if len(df) else 12)
            ws.set_column(i, i, min(max_w, 55))
        ws.freeze_panes(1, 0)
        if len(df.columns) > 1:
            ws.autofilter(0, 0, len(df), len(df.columns) - 1)
        ws.write_comment(0, 0, description)

    for df, sname, color, desc in SHEET_CONFIGS:
        styled_sheet(df, sname, color, desc)
        print(f"  Written: {sname} ({len(df)} rows)")

print(f"\nDone. Output: {OUTPUT_FILE}")