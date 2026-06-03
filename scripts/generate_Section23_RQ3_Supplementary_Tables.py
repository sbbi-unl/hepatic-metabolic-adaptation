"""
generate_Section23_RQ3_Supplementary_Tables.py
===============================================
Generates: RQ3_Section23_Supplementary_Tables.xlsx
           (also saved as Section23_Supplementary_Tables.xlsx)

Input files (loaded dynamically from their respective subfolders):
    phase2_contribution_analysis.csv
    all_strain_contributions.csv
    conservation_analysis.csv
    statistical_tests.csv
    phase3_reaction_attribution.csv
    phase4_pathway_attribution.csv
    driver_consistency.csv
    lineage_attribution.csv
    location_attribution.csv
    function_attribution.csv
    phase1_bulk_cellular_overlap.csv

Significance criterion (RQ3):
    Threshold-based: meets >= 2 of 3 criteria:
        (1) |fold_change| > 1.5
        (2) |flux_change| > 0.1  mmol·gDW⁻¹·h⁻¹
        (3) relative_magnitude > 0.5
    Column: significant == True

Sheets produced:
    SA — Cell-type contribution scores (single-condition)
    SB — Cross-strain LEC contributions (9 strains × cell types)
    SC — Conservation analysis (CV, min/max strain per cell type)
    SD — Per-cell-type significant reactions (response rates)
    SE — Reaction-level attribution: category and primary driver
    SF — Pathway-level attribution summary
    SG — Driver consistency across strains
    SH — Hierarchical attribution (lineage / location / function)
    SI — Bulk-cellular overlap (validation)
"""

import os, sys
import pandas as pd
import numpy as np
import shutil
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
# Base directory for inputs (assuming script runs from project root)
BASE_DIR    = Path("Processing_outputs")

# Output directory
OUTPUT_DIR  = BASE_DIR / "Supplementary_Tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "RQ3_Section23_Supplementary_Tables.xlsx"
OUTPUT_ALIAS = OUTPUT_DIR / "Section23_Supplementary_Tables.xlsx"

# ── Helpers ────────────────────────────────────────────────────────────────────
def load(rel_path):
    """Loads a CSV relative to the BASE_DIR."""
    path = BASE_DIR / rel_path
    if not path.exists():
        sys.exit(f"ERROR: Required file not found: {path}\nEnsure you are running this from the main project folder.")
    return pd.read_csv(path)

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading RQ3 data files from subfolders...")

# Integration Phase 1-4 tables
contrib    = load("Step_3_RQ3/Step_3b_results_rq1_rq3_integration/tables/phase2_contribution_analysis.csv")
rxn_attr   = load("Step_3_RQ3/Step_3b_results_rq1_rq3_integration/tables/phase3_reaction_attribution.csv")
path_attr  = load("Step_3_RQ3/Step_3b_results_rq1_rq3_integration/tables/phase4_pathway_attribution.csv")
overlap    = load("Step_3_RQ3/Step_3b_results_rq1_rq3_integration/tables/phase1_bulk_cellular_overlap.csv")

# Strain multi-background tables
all_str    = load("Step_3_RQ3/Step_3b_results_rq2_rq3_integration/tables/all_strain_contributions.csv")
conserv    = load("Step_3_RQ3/Step_3b_results_rq2_rq3_integration/tables/conservation_analysis.csv")
drv_cons   = load("Step_3_RQ3/Step_3b_results_rq2_rq3_integration/tables/driver_consistency.csv")

# Statistics
stat_tests = load("Step_3_RQ3/statistics/statistical_tests.csv")

# Hierarchical Analysis
lineage    = load("Step_3_RQ3/Hierarchical_Analysis/lineage_attribution.csv")
location   = load("Step_3_RQ3/Hierarchical_Analysis/location_attribution.csv")
function_  = load("Step_3_RQ3/Hierarchical_Analysis/function_attribution.csv")


# ── SA: Cell-type contribution (single-condition) ─────────────────────────────
sa = contrib.copy()
sa = sa.sort_values("contribution_percent", ascending=False)
sa.columns = [c.replace("_", " ").title() for c in sa.columns]
# Add per-cell index: contribution_score / cell_abundance
sa_raw = contrib.copy()
sa_raw["per_abundance_index"] = (
    sa_raw["contribution_score"] / sa_raw["cell_abundance"].replace(0, np.nan)
)
sa_raw = sa_raw.sort_values("contribution_percent", ascending=False)
# Build final SA table
sa_final = sa_raw[[
    "cell_type", "cell_abundance", "n_significant", "n_total",
    "response_rate", "mean_flux_change", "contribution_score",
    "contribution_percent", "per_abundance_index"
]].copy()
sa_final.columns = [
    "Cell_Type", "Cell_Abundance_Fraction", "N_Significant_Reactions",
    "N_Total_Reactions", "Response_Rate", "Mean_Flux_Change",
    "Contribution_Score", "Contribution_Percent", "Per_Abundance_Index"
]
print(f"  SA: {len(sa_final)} cell types")

# ── SB: Cross-strain contributions (9 strains) ───────────────────────────────
sb = all_str.copy()
sb = sb.sort_values(["cell_type", "strain"])
sb.columns = [c.replace("_"," ").title() for c in sb.columns]
print(f"  SB: {len(sb)} rows (strains × cell types)")

# Pivot: rows=cell_type, cols=strain, values=contribution_percent
sb_pivot = all_str.pivot_table(
    index="cell_type", columns="strain",
    values="contribution_percent", aggfunc="first"
).reset_index()
sb_pivot.insert(1, "Mean_Contribution_Pct",
    all_str.groupby("cell_type")["contribution_percent"].mean().values)
sb_pivot.insert(2, "SD_Contribution_Pct",
    all_str.groupby("cell_type")["contribution_percent"].std().values)
sb_pivot = sb_pivot.sort_values("Mean_Contribution_Pct", ascending=False)
print(f"  SB pivot: {sb_pivot.shape}")

# ── SC: Conservation analysis ─────────────────────────────────────────────────
sc = conserv.copy()
sc.columns = [
    "Cell_Type", "Mean_Contribution_Pct", "SD_Contribution_Pct",
    "CV_Coefficient_of_Variation", "Min_Strain", "Min_Value_Pct",
    "Max_Strain", "Max_Value_Pct"
]
sc = sc.sort_values("CV_Coefficient_of_Variation", ascending=False)
print(f"  SC: {len(sc)} rows")

# ── SD: Per-cell-type significant reactions ───────────────────────────────────
sig_tests = stat_tests[stat_tests["significant"] == True].copy()
n_total_rxns = stat_tests["reaction_id"].nunique()

per_cell_sig = (
    sig_tests.groupby("cell_type")["reaction_id"]
    .nunique()
    .reset_index()
    .rename(columns={"reaction_id": "N_Significant_Reactions"})
)
per_cell_total = (
    stat_tests.groupby("cell_type")["reaction_id"]
    .nunique()
    .reset_index()
    .rename(columns={"reaction_id": "N_Total_Reactions"})
)
sd = per_cell_sig.merge(per_cell_total, on="cell_type", how="outer").fillna(0)
sd["Response_Rate_Pct"] = sd["N_Significant_Reactions"] / sd["N_Total_Reactions"] * 100
sd["Pct_of_562_Union"]  = sd["N_Significant_Reactions"] / 562 * 100

# Get Chow cell counts from aggregation summary if available
try:
    agg = load("Step_3_RQ3/aggregation_summary.csv")
    chow_counts = agg[agg["condition"]=="Chow"][["cell_type","n_cells"]].rename(
        columns={"n_cells":"N_Cells_Chow"})
    sd = sd.merge(chow_counts, on="cell_type", how="left")
    sd["Sig_Per_Cell_Chow"] = (
        sd["N_Significant_Reactions"] / sd["N_Cells_Chow"].replace(0, np.nan)
    )
except:
    pass

sd = sd.sort_values("N_Significant_Reactions", ascending=False)
print(f"  SD: {len(sd)} cell types with significant reactions")

# ── SE: Reaction-level attribution ───────────────────────────────────────────
se = rxn_attr.copy()
se.columns = [
    "Reaction_ID", "Reaction_Name", "Subsystem", "Attribution_Category",
    "N_Cell_Types_Driving", "Primary_Driver", "Secondary_Drivers", "All_Cell_Types"
]
se = se.sort_values(["Subsystem", "Primary_Driver"])
print(f"  SE: {len(se)} reactions with attribution data")

# ── SF: Pathway-level attribution ─────────────────────────────────────────────
sf = path_attr.copy()
sf.columns = [
    "Pathway", "N_Reactions", "Primary_Driver", "Primary_Driver_Pct",
    "Secondary_Drivers", "N_Unique_Driver_Reactions", "N_Cooperative_Reactions",
    "N_Multicellular_Reactions", "N_Non_Cellular_Reactions"
]
sf = sf.sort_values("Primary_Driver_Pct", ascending=False)
print(f"  SF: {len(sf)} pathways")

# ── SG: Driver consistency across strains ─────────────────────────────────────
sg = drv_cons.copy()
sg.columns = [
    "Primary_Driver", "N_Strains_Present",
    "Total_Reactions_as_Primary_Driver",
    "Mean_Reactions_Per_Strain", "Consistency_Score"
]
sg = sg.sort_values("Consistency_Score", ascending=False)
print(f"  SG: {len(sg)} primary drivers")

# ── SH: Hierarchical attribution ─────────────────────────────────────────────
# Combine lineage, location, function
lineage_c  = lineage.copy();  lineage_c.insert(0, "Framework", "Lineage")
location_c = location.copy(); location_c.insert(0, "Framework", "Location")
function_c = function_.copy();function_c.insert(0, "Framework", "Function")

lineage_c.columns  = ["Framework", "Group", "Contribution_Percent"]
location_c.columns = ["Framework", "Group", "Contribution_Percent"]
function_c.columns = ["Framework", "Group", "Contribution_Percent"]

sh = pd.concat([lineage_c, location_c, function_c], ignore_index=True)
sh["Contribution_Percent"] = sh["Contribution_Percent"].round(2)
print(f"  SH: {len(sh)} rows (lineage+location+function)")

# ── SI: Bulk-cellular overlap ─────────────────────────────────────────────────
si = overlap.copy()
si.columns = [
    "Cell_Type", "N_Bulk_Significant", "N_Cell_Significant",
    "N_Overlap", "N_Bulk_Only", "N_Cell_Only",
    "Overlap_Pct", "Sensitivity", "Precision", "F1_Score", "Cell_Abundance"
]
si = si.sort_values("F1_Score", ascending=False)
print(f"  SI: {len(si)} cell types in validation")

# ── Write Excel ────────────────────────────────────────────────────────────────
print(f"\nWriting {OUTPUT_FILE} ...")

SHEET_CONFIGS = [
    (sa_final,  "SA_CellType_Contributions",     "#2F4F8F", "Cell type contribution to hepatic flux responses"),
    (sb_pivot,  "SB_CrossStrain_Contributions",  "#4B0082", "LEC/cell-type contributions across 9 inbred strains"),
    (sc,        "SC_Conservation_Analysis",       "#006400", "Cross-strain conservation of cell-type contributions"),
    (sd,        "SD_SignificantReactions_PerCell", "#8B0000","Per-cell-type significant reaction counts and response rates"),
    (se,        "SE_Reaction_Attribution",        "#00008B", "Reaction-level cellular attribution (category + driver)"),
    (sf,        "SF_Pathway_Attribution",         "#2F4F4F", "Pathway-level cellular attribution"),
    (sg,        "SG_Driver_Consistency",          "#8B4513", "Primary driver consistency across strains"),
    (sh,        "SH_Hierarchical_Attribution",    "#1F497D", "Hierarchical attribution: lineage / location / function"),
    (si,        "SI_Bulk_Cellular_Overlap",       "#4B0082", "Bulk vs single-cell flux validation overlap"),
]

with pd.ExcelWriter(OUTPUT_FILE, engine="xlsxwriter") as writer:
    wb = writer.book

    def styled_sheet(df, sheet_name, header_color, description):
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        ws = writer.sheets[sheet_name]
        hf = wb.add_format({
            "bold": True, "bg_color": header_color, "font_color": "white",
            "border": 1, "text_wrap": True, "valign": "vcenter"
        })
        # Description row above header (row 0 = description, row 1 = header)
        # Actually keep simple: just header row
        for i, col in enumerate(df.columns):
            ws.write(0, i, col, hf)
            max_w = max(
                len(str(col)) + 2,
                (df[col].astype(str).str.len().max() + 2) if len(df) > 0 else 12
            )
            ws.set_column(i, i, min(max_w, 55))
        ws.freeze_panes(1, 0)
        if len(df.columns) > 1:
            ws.autofilter(0, 0, len(df), len(df.columns) - 1)
        # Add description as a cell comment on A1
        ws.write_comment(0, 0, description)

    for df, sname, color, desc in SHEET_CONFIGS:
        styled_sheet(df, sname, color, desc)
        print(f"  Written: {sname} ({len(df)} rows)")

print(f"\nDone. Output: {OUTPUT_FILE}")

# Save alias
shutil.copy(OUTPUT_FILE, OUTPUT_ALIAS)
print(f"Alias saved: {OUTPUT_ALIAS}")