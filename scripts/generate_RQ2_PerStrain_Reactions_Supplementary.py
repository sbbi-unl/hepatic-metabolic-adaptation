"""
generate_RQ2_PerStrain_Reactions_Supplementary.py
==================================================
Generates: RQ2_PerStrain_Reactions_Supplementary.xlsx

Input files (expected in nested strain folders):
    Step_2_RQ2/results_129S1SvImJ_GSE182668/cytoscape_edges/edges_HFD_vs_SCD.csv
    Step_2_RQ2/results_AJ_GSE182668/cytoscape_edges/edges_HFD_vs_SCD.csv
    ...

Significance criterion (RQ2):
    |Diff(HFD-SCD)| >= 0.2  mmol·gDW⁻¹·h⁻¹  (threshold-based, no FDR)

Sheets produced:
    S1  — 129S1/SvImJ   significant reactions
    S2  — A/J            significant reactions
    S3  — C57BL/6J       significant reactions
    S4  — CAST/EiJ       significant reactions
    S5  — DBA/2J         significant reactions
    S6  — NOD/ShiLtJ     significant reactions
    S7  — NZO/HlLtJ      significant reactions
    S8  — PWK/PhJ        significant reactions
    S9  — WSB/EiJ        significant reactions
    S10 — Universal conserved reactions (present in ALL 9 strains)
    S11 — Summary: per-strain reaction counts + conservation tier
    S12 — Cross-strain reaction presence matrix
"""

import os, sys
import pandas as pd
import numpy as np
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
# Pointing to the main RQ2 outputs directory
DATA_DIR    = Path("Processing_outputs/Step_2_RQ2")
OUTPUT_DIR = Path("Processing_outputs/Supplementary_Tables")

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "RQ2_PerStrain_Reactions_Supplementary.xlsx"

FLUX_THRESHOLD = 0.2   # |Diff(HFD-SCD)| >= 0.2 mmol·gDW⁻¹·h⁻¹

STRAINS = {
    "129S1SvImJ": "129S1/SvImJ",
    "AJ":         "A/J",
    "C57BL6J":    "C57BL/6J",
    "CASTEiJ":    "CAST/EiJ",
    "DBA2J":      "DBA/2J",
    "NODShiLtJ":  "NOD/ShiLtJ",
    "NZOHlLtJ":   "NZO/HlLtJ",
    "PWKPhJ":     "PWK/PhJ",
    "WSBEiJ":     "WSB/EiJ",
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def load(fname):
    path = DATA_DIR / fname
    if not path.exists():
        sys.exit(f"ERROR: Required file not found: {path}\nMake sure you are running this script from your main project folder.")
    return pd.read_csv(path)

# ── Load all strain edge files ─────────────────────────────────────────────────
print("Loading RQ2 strain edge files...")
strain_dfs = {}
for key, label in STRAINS.items():
    # Construct the nested path specifically for each strain
    fname = f"results_{key}_GSE182668/cytoscape_edges/edges_HFD_vs_SCD.csv"
    df = load(fname)
    
    # Already filtered to |Diff| >= threshold by the pipeline,
    # but enforce explicitly for reproducibility
    df = df[df["Diff(HFD-SCD)"].abs() >= FLUX_THRESHOLD].copy()
    df["Strain"] = label
    df["Direction"] = df["Diff(HFD-SCD)"].apply(lambda x: "Up" if x > 0 else "Down")
    strain_dfs[key] = df
    print(f"  {label:<15}: {df['ReactionID'].nunique()} unique reactions, {len(df)} edges")

# ── Build unique reaction sets per strain ──────────────────────────────────────
per_strain_rxns = {k: set(v["ReactionID"].unique()) for k, v in strain_dfs.items()}

# Universal conserved: in ALL 9 strains
universal = set.intersection(*per_strain_rxns.values())

# Conservation tiers
def count_strains(rxn_id):
    return sum(1 for rxns in per_strain_rxns.values() if rxn_id in rxns)

all_unique_rxns = set.union(*per_strain_rxns.values())
print(f"\n  Total unique reactions (any strain): {len(all_unique_rxns)}")
print(f"  Universal (all 9 strains): {len(universal)}")
print(f"  Not conserved across all: {len(all_unique_rxns) - len(universal)}")

# ── Build universal conserved table ───────────────────────────────────────────
universal_rows = []
for rxn_id in universal:
    # Get reaction metadata from first strain that has it
    for df in strain_dfs.values():
        row_df = df[df["ReactionID"] == rxn_id]
        if len(row_df) > 0:
            row = row_df.iloc[0]
            # Get direction agreement
            directions = []
            for sdf in strain_dfs.values():
                srow = sdf[sdf["ReactionID"] == rxn_id]
                if len(srow) > 0:
                    directions.append("Up" if srow.iloc[0]["Diff(HFD-SCD)"] > 0 else "Down")
            n_up = directions.count("Up")
            n_dn = directions.count("Down")
            n_strains = len(directions)
            dominant_dir = "Up" if n_up >= n_dn else "Down"
            pct_agree = max(n_up, n_dn) / n_strains * 100
            universal_rows.append({
                "ReactionID":     rxn_id,
                "ReactionName":   row["ReactionName"],
                "Subsystem":      row["Subsystem"],
                "N_Strains":      n_strains,
                "N_Up":           n_up,
                "N_Down":         n_dn,
                "Dominant_Direction": dominant_dir,
                "Pct_DirectionalAgreement": round(pct_agree, 1),
            })
            break

universal_df = pd.DataFrame(universal_rows).sort_values("Subsystem")

# ── Build summary table ────────────────────────────────────────────────────────
summary_rows = []
for key, label in STRAINS.items():
    rxns = per_strain_rxns[key]
    univ_in_strain = len(rxns & universal)
    up_rxns   = strain_dfs[key][strain_dfs[key]["Direction"]=="Up"]["ReactionID"].nunique()
    down_rxns = strain_dfs[key][strain_dfs[key]["Direction"]=="Down"]["ReactionID"].nunique()
    summary_rows.append({
        "Strain":                      label,
        "N_Unique_Reactions":          len(rxns),
        "N_Up_Regulated":              up_rxns,
        "N_Down_Regulated":            down_rxns,
        "N_Universal_Conserved":       univ_in_strain,
        "N_Strain_Variable":           len(rxns) - univ_in_strain,
        "Pct_In_Universal":            round(univ_in_strain/len(rxns)*100, 1) if rxns else 0,
    })
summary_df = pd.DataFrame(summary_rows)

# Add overall row
all_union = len(all_unique_rxns)
summary_df.loc[len(summary_df)] = {
    "Strain": "UNION (any strain)",
    "N_Unique_Reactions":    all_union,
    "N_Up_Regulated":        "",
    "N_Down_Regulated":      "",
    "N_Universal_Conserved": len(universal),
    "N_Strain_Variable":     all_union - len(universal),
    "Pct_In_Universal":      round(len(universal)/all_union*100, 1),
}

# ── Build cross-strain presence matrix ────────────────────────────────────────
print("\nBuilding cross-strain presence matrix...")
matrix_rows = []
for rxn_id in sorted(all_unique_rxns):
    row = {"ReactionID": rxn_id}
    # Get metadata
    for df in strain_dfs.values():
        m = df[df["ReactionID"]==rxn_id]
        if len(m) > 0:
            row["ReactionName"] = m.iloc[0]["ReactionName"]
            row["Subsystem"]    = m.iloc[0]["Subsystem"]
            break
    for key, label in STRAINS.items():
        sdf = strain_dfs[key]
        srow = sdf[sdf["ReactionID"]==rxn_id]
        if len(srow) > 0:
            row[label] = "1" if srow.iloc[0]["Diff(HFD-SCD)"] > 0 else "-1"
        else:
            row[label] = "0"
    row["N_Strains_Present"] = count_strains(rxn_id)
    row["Conservation_Tier"] = (
        "Universal"  if row["N_Strains_Present"] == 9 else
        "Majority"   if row["N_Strains_Present"] >= 5 else
        "Minority"   if row["N_Strains_Present"] >= 2 else
        "Unique"
    )
    matrix_rows.append(row)

matrix_df = pd.DataFrame(matrix_rows).sort_values(
    ["N_Strains_Present", "Subsystem"], ascending=[False, True]
)

# ── Write Excel ────────────────────────────────────────────────────────────────
print(f"\nWriting {OUTPUT_FILE.name} ...")
STRAIN_COLORS = {
    "129S1SvImJ": "#8B0000", "AJ": "#00008B",   "C57BL6J": "#006400",
    "CASTEiJ":    "#FF8C00", "DBA2J": "#800080", "NODShiLtJ": "#008B8B",
    "NZOHlLtJ":   "#8B4513", "PWKPhJ": "#2F4F4F","WSBEiJ": "#4B0082",
}

with pd.ExcelWriter(OUTPUT_FILE, engine="xlsxwriter") as writer:
    wb = writer.book

    def styled_sheet(df, sheet_name, header_color="#1F497D"):
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
        ws.autofilter(0, 0, len(df), len(df.columns) - 1)

    # S1–S9: Per-strain significant reactions
    for i, (key, label) in enumerate(STRAINS.items(), 1):
        df = strain_dfs[key].sort_values("Diff(HFD-SCD)")
        sheet_name = f"S{i}_{key[:12]}"
        styled_sheet(df, sheet_name, STRAIN_COLORS[key])

    # S10: Universal conserved reactions
    styled_sheet(universal_df, "S10_Universal_Conserved", "#2F4F8F")

    # S11: Summary
    styled_sheet(summary_df, "S11_Summary", "#1F497D")

    # S12: Cross-strain presence matrix
    styled_sheet(matrix_df, "S12_Presence_Matrix", "#1F497D")

print(f"Done. Output: {OUTPUT_FILE}")
print(f"  Sheets: S1-S9 (per strain) + S10 (universal) + S11 (summary) + S12 (matrix)")