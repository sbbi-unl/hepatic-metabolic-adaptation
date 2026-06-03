"""
generate_Section33_Supplementary_Tables.py
==========================================
Generates: Section33_Supplementary_Tables.xlsx

Input files (loaded dynamically from their respective subfolders):
    phase2_contribution_analysis.csv
    all_strain_contributions.csv
    lineage_attribution.csv
    location_attribution.csv
    function_attribution.csv
    conservation_analysis.csv

Sheets produced:
    SL1 — Cross-RQ cell-type contribution table
           (single-condition + cross-strain mean ± SD + per-abundance index)
    SL2 — Lineage / Location / Function hierarchical attribution
    SL3 — Cross-strain contribution matrix (rows=cell types, cols=strains)
"""

import os, sys
import pandas as pd
import numpy as np
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
# Base directory for inputs (assuming script runs from project root)
BASE_DIR    = Path("Processing_outputs")

# Output directory
OUTPUT_DIR  = BASE_DIR / "Supplementary_Tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "Section33_Supplementary_Tables.xlsx"

# ── Helpers ────────────────────────────────────────────────────────────────────
def load(rel_path):
    """Loads a CSV relative to the BASE_DIR."""
    path = BASE_DIR / rel_path
    if not path.exists():
        sys.exit(f"ERROR: Required file not found: {path}\nEnsure you are running this from the main project folder.")
    return pd.read_csv(path)

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading data files for Section 3.3 from subfolders...")
contrib  = load("Step_3_RQ3/Step_3b_results_rq1_rq3_integration/tables/phase2_contribution_analysis.csv")
all_str  = load("Step_3_RQ3/Step_3b_results_rq2_rq3_integration/tables/all_strain_contributions.csv")
conserv  = load("Step_3_RQ3/Step_3b_results_rq2_rq3_integration/tables/conservation_analysis.csv")

lineage  = load("Step_3_RQ3/Hierarchical_Analysis/lineage_attribution.csv")
location = load("Step_3_RQ3/Hierarchical_Analysis/location_attribution.csv")
function = load("Step_3_RQ3/Hierarchical_Analysis/function_attribution.csv")

# ── SL1: Comprehensive Cell-Type Contribution ──────────────────────────────────
print("Building SL1: Comprehensive cell-type contributions...")
# 1. Base single-condition stats (from phase2)
sl1 = contrib[[
    "cell_type", "cell_abundance", "n_significant", "n_total",
    "response_rate", "contribution_score", "contribution_percent"
]].copy()

# 2. Add per-abundance index
sl1["per_abundance_index"] = (
    sl1["contribution_score"] / sl1["cell_abundance"].replace(0, np.nan)
)

# 3. Add cross-strain stats (from all_strain_contributions)
mean_str = all_str.groupby("cell_type")["contribution_percent"].mean().rename("CrossStrain_Mean_Pct")
sd_str   = all_str.groupby("cell_type")["contribution_percent"].std().rename("CrossStrain_SD_Pct")
sl1 = sl1.merge(mean_str, on="cell_type", how="left")
sl1 = sl1.merge(sd_str, on="cell_type", how="left")

# 4. Fold vs Hepatocytes (reference)
# Hepatocytes contribution in single-condition
hep_row = sl1[sl1["cell_type"].str.contains("Hepatocyte", case=False, na=False)]
if len(hep_row) > 0:
    hep_pct = hep_row.iloc[0]["contribution_percent"]
    if hep_pct > 0:
        sl1["Fold_vs_Hepatocytes"] = sl1["contribution_percent"] / hep_pct
    else:
        sl1["Fold_vs_Hepatocytes"] = np.nan
else:
    sl1["Fold_vs_Hepatocytes"] = np.nan

sl1 = sl1.sort_values("contribution_percent", ascending=False)
sl1.columns = [
    "Cell_Type", "Abundance_Fraction", "N_Significant_Reactions", "N_Total_Reactions",
    "Response_Rate", "Attribution_Score", "SingleStrain_Contribution_Pct",
    "Per_Abundance_Index", "MultiStrain_Mean_Pct", "MultiStrain_SD_Pct", "Fold_vs_Hepatocytes"
]
print(f"  SL1: {len(sl1)} cell types")

# ── SL2: Hierarchical Attribution ──────────────────────────────────────────────
print("Building SL2: Hierarchical attribution...")
lineage_c  = lineage.copy();  lineage_c.insert(0, "Hierarchy", "Lineage")
location_c = location.copy(); location_c.insert(0, "Hierarchy", "Location")
function_c = function.copy(); function_c.insert(0, "Hierarchy", "Function")

sl2 = pd.concat([lineage_c, location_c, function_c], ignore_index=True)
sl2.columns = ["Hierarchy", "Group", "Contribution_Percent"]
sl2["Contribution_Percent"] = sl2["Contribution_Percent"].round(2)
print(f"  SL2: {len(sl2)} hierarchical groups")

# ── SL3: Cross-Strain Matrix ───────────────────────────────────────────────────
print("Building SL3: Cross-strain contribution matrix...")
# Pivot table: rows = cell_type, columns = strain
sl3_pivot = all_str.pivot_table(
    index="cell_type", columns="strain",
    values="contribution_percent", aggfunc="first"
).reset_index()

# Extract exactly the target columns from conservation_analysis.csv
conserv_sub = conserv[["cell_type", "mean_contribution", "std_contribution", "cv"]].rename(
    columns={
        "mean_contribution": "Mean_Pct", 
        "std_contribution": "SD_Pct", 
        "cv": "CV"
    }
)

sl3_pivot = sl3_pivot.merge(conserv_sub, on="cell_type", how="left")
sl3_pivot = sl3_pivot.sort_values("Mean_Pct", ascending=False)
print(f"  SL3: {len(sl3_pivot)} cell types across strains")

# ── Write Excel ────────────────────────────────────────────────────────────────
print(f"\nWriting {OUTPUT_FILE.name} ...")

SHEET_CONFIGS = [
    (sl1, "SL1_CellType_Contributions", "#8B0000",
     "Comprehensive cell-type contribution table: single-condition, cross-strain mean±SD, per-abundance index, fold vs hepatocytes"),
    (sl2, "SL2_Hierarchical_Attribution","#4B0082",
     "Hierarchical attribution by lineage (Endothelial/Myeloid/Mesenchymal/Epithelial/Lymphoid), location, and function"),
    (sl3_pivot, "SL3_CrossStrain_Matrix","#006400",
     "Cell-type contribution percentages across all 9 inbred mouse strains plus mean, SD, and CV"),
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
            ws.set_column(i, i, min(max_w, 50))
        ws.freeze_panes(1, 0)
        if len(df.columns) > 1:
            ws.autofilter(0, 0, len(df), len(df.columns) - 1)
        ws.write_comment(0, 0, description)

    for df, sname, color, desc in SHEET_CONFIGS:
        styled_sheet(df, sname, color, desc)
        print(f"  Written: {sname} ({len(df)} rows)")

print(f"\nDone. Output: {OUTPUT_FILE}")