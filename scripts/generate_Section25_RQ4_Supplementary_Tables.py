"""
generate_Section25_RQ4_Supplementary_Tables.py
================================================
Generates: Section26_Supplementary_Tables.xlsx

Input files (loaded dynamically from their respective subfolders):
    flux_attribution_analysis.csv
    pathway_enrichment_results.csv
    compartment_enrichment_results.csv
    pathway_synergy_analysis.csv
    species_abundances.csv
    portal_metabolite_production.csv
    reaction_annotations.csv
    ND_SCD_flux_comparison.csv
    DD_HFD_flux_comparison.csv
    species_to_model_mapping.csv

Sheets produced:
    SK1 — All 3726 reactions: diet vs microbiome variance attribution
    SK2 — Diet-dominated reactions (dominant_driver == 'Diet*')
    SK3 — Microbiome-influenced reactions (variance_explained_microbiome > 0)
    SK4 — Pathway-level enrichment and synergy analysis
    SK5 — Compartment enrichment analysis
    SK6 — Species abundances and portal metabolites
    SK7 — ND/SCD vs DD/HFD flux comparisons (microbiome baseline)
    SK8 — Species to AGORA2 model mapping
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

OUTPUT_FILE = OUTPUT_DIR / "Section25_Supplementary_Tables.xlsx"

# ── Helpers ────────────────────────────────────────────────────────────────────
def load(rel_path):
    """Loads a CSV relative to the BASE_DIR."""
    path = BASE_DIR / rel_path
    if not path.exists():
        sys.exit(f"ERROR: Required file not found: {path}\nEnsure you are running this from the main project folder.")
    return pd.read_csv(path)

def load_optional(rel_path):
    """Loads an optional CSV relative to the BASE_DIR."""
    path = BASE_DIR / rel_path
    if not path.exists():
        print(f"  WARNING: Optional file not found: {path} — sheet will be empty")
        return pd.DataFrame()
    return pd.read_csv(path)

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading RQ4 data files from subfolders...")

flux_attr = load("Step_4_RQ4/03_attribution_analysis/flux_attribution_analysis.csv")
path_enr  = load("Step_4_RQ4/03_attribution_analysis/pathway_enrichment/pathway_enrichment_results.csv")
comp_enr  = load("Step_4_RQ4/03_attribution_analysis/pathway_enrichment/compartment_enrichment_results.csv")
synergy   = load("Step_4_RQ4/03_attribution_analysis/pathway_enrichment/pathway_synergy_analysis.csv")
species   = load("Step_4_RQ4/01_community_modeling/species_abundances.csv")
portal    = load("Step_4_RQ4/01_community_modeling/portal_metabolite_production.csv")
rxn_annot = load("Step_4_RQ4/reaction_annotations.csv")

nd_flux   = load_optional("Step_4_RQ4/02_hepatic_integration/condition_ND_SCD/condition_ND_SCD/ND_SCD_flux_comparison.csv")
dd_flux   = load_optional("Step_4_RQ4/02_hepatic_integration/condition_DD_HFD/condition_DD_HFD/DD_HFD_flux_comparison.csv")
sp_map    = load_optional("Step_4_RQ4/01_community_modeling/species_to_model_mapping.csv")

# ── SK1: Full attribution table ───────────────────────────────────────────────
print("Building SK1: Full flux attribution...")
sk1 = flux_attr.merge(
    rxn_annot[["reaction_id","reaction_name","subsystem","compartment","compartment_readable"]],
    on="reaction_id", how="left"
)
# Reorder columns: identifiers first, then attribution metrics
id_cols = ["reaction_id","reaction_name","subsystem","compartment","compartment_readable"]
metric_cols = [c for c in flux_attr.columns if c != "reaction_id"]
sk1 = sk1[id_cols + metric_cols]
sk1 = sk1.sort_values("variance_explained_microbiome", ascending=False)
print(f"  SK1: {len(sk1)} reactions")

# ── SK2: Diet-dominated reactions ─────────────────────────────────────────────
print("Building SK2: Diet-dominated reactions...")
sk2 = sk1[sk1["dominant_driver"].str.startswith("Diet", na=False)].copy()
sk2 = sk2.sort_values("abs_diet_contribution", ascending=False)
print(f"  SK2: {len(sk2)} diet-dominated reactions")

# ── SK3: Microbiome-influenced reactions (VE_micro > 0) ──────────────────────
print("Building SK3: Microbiome-influenced reactions...")
sk3 = sk1[sk1["variance_explained_microbiome"] > 0].copy()
sk3 = sk3.sort_values("variance_explained_microbiome", ascending=False)

# Add summary stats
print(f"  SK3: {len(sk3)} microbiome-influenced reactions")
if len(sk3) > 0:
    print(f"    Mean VE microbiome: {sk3['variance_explained_microbiome'].mean():.2f}%")
    print(f"    Microbiome-dominant: {(sk3['dominant_driver']=='Microbiome').sum()}")

# ── SK4: Pathway enrichment + synergy ─────────────────────────────────────────
print("Building SK4: Pathway enrichment and synergy...")
# Merge enrichment and synergy on subsystem
sk4 = path_enr.merge(
    synergy.rename(columns={"subsystem":"subsystem"}),
    on="subsystem", how="outer", suffixes=("_enrichment","_synergy")
)
# Rename for clarity
sk4.columns = [
    "Subsystem", "N_Reactions_Enr", "N_Significant_Enr", "Expected_Enr",
    "Enrichment_Ratio", "P_Value_Enr", "Diet_Dominated_Pct", "Microbiome_Dominated_Pct",
    "Synergistic_Pct_Enr", "Mean_Diet_Variance", "Mean_Microbiome_Variance",
    "P_Adjusted_Enr", "Significant_Enr",
    "N_Reactions_Syn", "N_Synergistic", "N_Antagonistic",
    "Synergistic_Pct", "Antagonistic_Pct", "Pathway_Class"
]
sk4 = sk4.sort_values("Enrichment_Ratio", ascending=False)
print(f"  SK4: {len(sk4)} pathways")

# ── SK5: Compartment enrichment ───────────────────────────────────────────────
print("Building SK5: Compartment enrichment...")
sk5 = comp_enr.copy()
sk5.columns = [
    "Compartment", "N_Reactions", "N_Significant", "Expected",
    "Enrichment_Ratio", "P_Value",
    "Mean_Microbiome_Contribution", "Mean_Diet_Contribution",
    "Mean_Microbiome_Variance", "Large_Microbiome_Effects_Pct",
    "P_Adjusted", "Significant"
]
# Add readable compartment if missing
if "Compartment_Readable" not in sk5.columns:
    comp_readable = rxn_annot.groupby("compartment")["compartment_readable"].first().to_dict()
    sk5["Compartment_Readable"] = sk5["Compartment"].map(comp_readable)
sk5 = sk5.sort_values("Enrichment_Ratio", ascending=False)
print(f"  SK5: {len(sk5)} compartments")

# ── SK6: Species abundances + portal metabolites ──────────────────────────────
print("Building SK6: Species abundances and portal metabolites...")
# Species abundance with fold change
sp = species.copy()
sp["Log2FC_DD_vs_ND"] = np.log2(
    sp["DD_HFD_abundance"].replace(0, np.nan) /
    sp["ND_SCD_abundance"].replace(0, np.nan)
)
sp["Direction"] = sp["Log2FC_DD_vs_ND"].apply(
    lambda x: "Enriched in DD/HFD" if x > 0
    else ("Depleted in DD/HFD" if x < 0 else "No change")
    if not pd.isna(x) else "No change"
)
sp = sp.sort_values("DD_HFD_abundance", ascending=False)
sp.columns = [
    "Species", "Abundance_ND_SCD", "Abundance_DD_HFD",
    "Log2FC_DD_vs_ND", "Direction"
]

# Portal metabolites
port = portal.copy()
port.columns = ["Condition", "Metabolite", "Metabolite_ID", "Flux_mmol_per_gDW_per_hr", "Importance"]

print(f"  SK6: {len(sp)} species, {len(port)} portal metabolite rows")

# ── SK7: ND/SCD vs DD/HFD flux comparisons ────────────────────────────────────
print("Building SK7: Flux comparisons...")
if len(nd_flux) > 0 and len(dd_flux) > 0:
    nd_renamed = nd_flux.rename(columns={
        "flux_baseline":     "ND_Baseline_Flux",
        "flux_microbiome":   "ND_Microbiome_Flux",
        "flux_delta":        "ND_Delta",
        "flux_delta_pct":    "ND_Delta_Pct",
        "microbiome_attributable": "ND_Microbiome_Attributable"
    })
    dd_renamed = dd_flux.rename(columns={
        "flux_baseline":     "DD_Baseline_Flux",
        "flux_microbiome":   "DD_Microbiome_Flux",
        "flux_delta":        "DD_Delta",
        "flux_delta_pct":    "DD_Delta_Pct",
        "microbiome_attributable": "DD_Microbiome_Attributable"
    })
    sk7 = nd_renamed.merge(
        dd_renamed[["reaction_id","DD_Baseline_Flux","DD_Microbiome_Flux",
                    "DD_Delta","DD_Delta_Pct","DD_Microbiome_Attributable"]],
        on="reaction_id", how="outer"
    )
    sk7 = sk7.merge(rxn_annot[["reaction_id","reaction_name","subsystem"]], 
                    on="reaction_id", how="left")
    sk7 = sk7.sort_values("DD_Delta", ascending=False, key=abs)
else:
    sk7 = pd.DataFrame({"Note": ["ND_SCD_flux_comparison.csv or DD_HFD_flux_comparison.csv not found"]})
print(f"  SK7: {len(sk7)} reactions")

# ── SK8: Species to model mapping ─────────────────────────────────────────────
sk8 = sp_map.copy() if len(sp_map) > 0 else pd.DataFrame({"Note": ["File not found"]})
if "species" in sk8.columns:
    sk8.columns = ["Species", "AGORA2_Model_File"]
print(f"  SK8: {len(sk8)} species-model mappings")

# ── Write Excel ────────────────────────────────────────────────────────────────
print(f"\nWriting {OUTPUT_FILE.name} ...")

SHEET_CONFIGS = [
    (sk1, "SK1_Full_Attribution",        "#1F497D",
     "Variance attribution of all 3726 reactions: diet vs microbiome contributions"),
    (sk2, "SK2_Diet_Dominated",          "#8B0000",
     "Reactions where diet is the dominant driver of flux change"),
    (sk3, "SK3_Microbiome_Influenced",   "#006400",
     "Reactions with detectable microbiome contribution (variance_explained_microbiome > 0)"),
    (sk4, "SK4_Pathway_Enrichment_Syn",  "#4B0082",
     "Pathway enrichment (hypergeometric) and synergy/antagonism classification"),
    (sk5, "SK5_Compartment_Enrichment",  "#2F4F4F",
     "Compartment-level enrichment of microbiome-influenced reactions"),
    (sk6_sp := None, None, None, None),   # placeholder — write separately below
    (sk7, "SK7_Flux_Comparisons",        "#00008B",
     "ND/SCD vs DD/HFD hepatic flux comparisons (microbiome contribution)"),
    (sk8, "SK8_Species_Model_Map",       "#8B4513",
     "Mapping of gut microbiome species to AGORA2 metabolic reconstruction files"),
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

    for df, sname, color, desc in [
        (sk1, "SK1_Full_Attribution",       "#1F497D", "Variance attribution: all 3726 reactions"),
        (sk2, "SK2_Diet_Dominated",         "#8B0000", "Diet-dominated reactions"),
        (sk3, "SK3_Microbiome_Influenced",  "#006400", "Microbiome-influenced reactions (VE_micro > 0)"),
        (sk4, "SK4_Pathway_Enrichment_Syn", "#4B0082", "Pathway enrichment and synergy"),
        (sk5, "SK5_Compartment_Enrichment", "#2F4F4F", "Compartment enrichment"),
    ]:
        styled_sheet(df, sname, color, desc)
        print(f"  Written: {sname} ({len(df)} rows)")

    # SK6: Split species and portal metabolites across rows in one sheet
    # Write species table starting at row 0, then portal table below
    sk6_sheet = "SK6_Species_Portal"
    sp.to_excel(writer, sheet_name=sk6_sheet, index=False, startrow=0)
    ws6 = writer.sheets[sk6_sheet]
    hf6 = wb.add_format({"bold":True,"bg_color":"#8B0000","font_color":"white","border":1})
    hf6b= wb.add_format({"bold":True,"bg_color":"#006400","font_color":"white","border":1})
    for i, col in enumerate(sp.columns):
        ws6.write(0, i, col, hf6)
        ws6.set_column(i, i, 25)
    sep_row = len(sp) + 2
    ws6.write(sep_row, 0, "Portal Metabolite Production", wb.add_format({"bold":True,"font_size":12}))
    port.to_excel(writer, sheet_name=sk6_sheet, index=False, startrow=sep_row+1)
    for i, col in enumerate(port.columns):
        ws6.write(sep_row+1, i, col, hf6b)
    ws6.freeze_panes(1, 0)
    print(f"  Written: SK6_Species_Portal ({len(sp)} species + {len(port)} portal rows)")

    for df, sname, color, desc in [
        (sk7, "SK7_Flux_Comparisons",  "#00008B", "ND/SCD vs DD/HFD flux comparisons"),
        (sk8, "SK8_Species_Model_Map", "#8B4513", "Species to AGORA2 model mapping"),
    ]:
        styled_sheet(df, sname, color, desc)
        print(f"  Written: {sname} ({len(df)} rows)")

print(f"\nDone. Output: {OUTPUT_FILE}")