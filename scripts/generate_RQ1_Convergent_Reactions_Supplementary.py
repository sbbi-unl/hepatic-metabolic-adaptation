"""
generate_RQ1_Convergent_Reactions_Supplementary.py
====================================================
Generates: RQ1_Convergent_Reactions_Supplementary.xlsx

Input files:
    reaction_stats_HFD_vs_SCD.csv
    reaction_stats_KD_vs_SCD.csv
    reaction_stats_WD_vs_SCD.csv
    subsystem_analysis_HFD_vs_SCD.csv
    subsystem_analysis_KD_vs_SCD.csv
    subsystem_analysis_WD_vs_SCD.csv
    pathway_enrichment_HFD_vs_SCD.csv
    pathway_enrichment_KD_vs_SCD.csv
    pathway_enrichment_WD_vs_SCD.csv

Significance criteria (RQ1):
    FDR (BH-corrected q-value) < 0.05  AND  |Cohen's d| > 0.5

Sheets produced:
    S1  — HFD vs SCD: All significant reactions
    S2  — KD  vs SCD: All significant reactions
    S3  — WD  vs SCD: All significant reactions
    S4  — Convergent reactions: significant in BOTH HFD and KD vs SCD
    S5  — KD-unique reactions: significant in KD but NOT HFD vs SCD
    S6  — WD-unique reactions: significant in WD but NOT HFD vs SCD
    S7  — Subsystem analysis: HFD vs SCD
    S8  — Subsystem analysis: KD  vs SCD
    S9  — Subsystem analysis: WD  vs SCD
    S10 — Pathway enrichment: HFD vs SCD
    S11 — Pathway enrichment: KD  vs SCD
    S12 — Pathway enrichment: WD  vs SCD
"""

import os, sys
import pandas as pd
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
# Pointing directly to the comprehensive analysis CSV folder
DATA_DIR   = Path("Processing_outputs/Step_1_RQ1/Comprehensive_Analysis/csv_outputs")
OUTPUT_DIR = Path("Processing_outputs/Supplementary_Tables")

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "RQ1_Convergent_Reactions_Supplementary.xlsx"

FDR_THRESHOLD   = 0.05
COHEN_THRESHOLD = 0.5

# ── Helpers ────────────────────────────────────────────────────────────────────
def load(fname):
    path = DATA_DIR / fname
    if not path.exists():
        sys.exit(f"ERROR: Required file not found: {path}\nMake sure you are running this script from your main project folder.")
    return pd.read_csv(path)

def significant(df, q_col="q_value", d_col="Cohen_d"):
    return df[(df[q_col] < FDR_THRESHOLD) & (df[d_col].abs() > COHEN_THRESHOLD)].copy()

def add_direction(df, d_col="Cohen_d"):
    df["Direction"] = df[d_col].apply(lambda x: "Up" if x > 0 else "Down")
    return df

def write_sheet(writer, df, sheet_name, col_widths=None):
    df.to_excel(writer, sheet_name=sheet_name, index=False)
    ws = writer.sheets[sheet_name]
    # Auto-width
    for i, col in enumerate(df.columns):
        width = max(len(str(col)) + 2,
                    df[col].astype(str).str.len().max() + 2 if len(df) > 0 else 10)
        ws.set_column(i, i, min(width, 60))

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading reaction statistics...")
rxn_hfd = load("reaction_stats_HFD_vs_SCD.csv")
rxn_kd  = load("reaction_stats_KD_vs_SCD.csv")
rxn_wd  = load("reaction_stats_WD_vs_SCD.csv")

print("Loading subsystem analyses...")
sub_hfd = load("subsystem_analysis_HFD_vs_SCD.csv")
sub_kd  = load("subsystem_analysis_KD_vs_SCD.csv")
sub_wd  = load("subsystem_analysis_WD_vs_SCD.csv")

print("Loading pathway enrichment results...")
enr_hfd = load("pathway_enrichment_HFD_vs_SCD.csv")
enr_kd  = load("pathway_enrichment_KD_vs_SCD.csv")
enr_wd  = load("pathway_enrichment_WD_vs_SCD.csv")

# ── Build significant reaction tables ─────────────────────────────────────────
sig_hfd = significant(rxn_hfd)
sig_kd  = significant(rxn_kd)
sig_wd  = significant(rxn_wd)

# Add direction
for df in [sig_hfd, sig_kd, sig_wd]:
    add_direction(df)

# S4: Convergent — HFD ∩ KD (both significant vs SCD)
hfd_ids = set(sig_hfd["ReactionID"])
kd_ids  = set(sig_kd["ReactionID"])
wd_ids  = set(sig_wd["ReactionID"])

convergent_ids = hfd_ids & kd_ids
conv = sig_hfd[sig_hfd["ReactionID"].isin(convergent_ids)].copy()
conv = conv.rename(columns={
    "HFD_mean": "HFD_MeanFlux", "SCD_mean": "SCD_MeanFlux_HFD",
    "MeanDiff": "Diff_HFD_SCD", "log2FC": "log2FC_HFD",
    "Cohen_d": "Cohen_d_HFD", "q_value": "q_HFD"
})
# Merge KD d and q
kd_sub = sig_kd[sig_kd["ReactionID"].isin(convergent_ids)][
    ["ReactionID", "Cohen_d", "q_value", "MeanDiff"]
].rename(columns={"Cohen_d":"Cohen_d_KD","q_value":"q_KD","MeanDiff":"Diff_KD_SCD"})
conv = conv.merge(kd_sub, on="ReactionID", how="left")
# Flag directional agreement
conv["Direction_HFD"] = conv["Cohen_d_HFD"].apply(lambda x: "Up" if x>0 else "Down")
conv["Direction_KD"]  = conv["Cohen_d_KD"].apply(lambda x: "Up" if x>0 else "Down")
conv["DirectionalAgreement"] = conv["Direction_HFD"] == conv["Direction_KD"]
conv = conv.sort_values("Cohen_d_HFD")

# S5: KD-unique (significant in KD but not HFD vs SCD)
kd_only = sig_kd[~sig_kd["ReactionID"].isin(hfd_ids)].copy().sort_values("Cohen_d")

# S6: WD-unique (significant in WD but not HFD vs SCD)
wd_only = sig_wd[~sig_wd["ReactionID"].isin(hfd_ids)].copy().sort_values("Cohen_d")

# Print summary
print(f"\n  Significant reactions (FDR<{FDR_THRESHOLD}, |d|>{COHEN_THRESHOLD}):")
print(f"    HFD vs SCD: {len(sig_hfd)}")
print(f"    KD  vs SCD: {len(sig_kd)}")
print(f"    WD  vs SCD: {len(sig_wd)}")
print(f"    Convergent (HFD∩KD): {len(convergent_ids)}")
print(f"    KD-unique:  {len(kd_only)}")
print(f"    WD-unique:  {len(wd_only)}")

# ── Write Excel ────────────────────────────────────────────────────────────────
print(f"\nWriting {OUTPUT_FILE.name} ...")
with pd.ExcelWriter(OUTPUT_FILE, engine="xlsxwriter") as writer:
    wb = writer.book
    # Formats
    hdr_fmt = wb.add_format({
        "bold": True, "bg_color": "#2F4F8F", "font_color": "white",
        "border": 1, "text_wrap": True, "valign": "vcenter"
    })
    up_fmt   = wb.add_format({"bg_color": "#FFE0E0"})
    down_fmt = wb.add_format({"bg_color": "#E0E8FF"})
    sig_fmt  = wb.add_format({"bg_color": "#E0FFE0"})
    num_fmt  = wb.add_format({"num_format": "0.0000"})
    pct_fmt  = wb.add_format({"num_format": "0.00%"})

    def styled_sheet(df, sheet_name, header_color="#2F4F8F"):
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

    # S1–S3: Per-comparison significant reactions
    for df, name, color in [
        (sig_hfd, "S1_HFD_vs_SCD_Significant", "#8B0000"),
        (sig_kd,  "S2_KD_vs_SCD_Significant",  "#00008B"),
        (sig_wd,  "S3_WD_vs_SCD_Significant",  "#006400"),
    ]:
        styled_sheet(df.sort_values("Cohen_d"), name, color)

    # S4: Convergent reactions
    styled_sheet(conv, "S4_Convergent_HFD_KD", "#4B0082")

    # S5–S6: Diet-unique reactions
    styled_sheet(kd_only, "S5_KD_Unique", "#00008B")
    styled_sheet(wd_only, "S6_WD_Unique", "#006400")

    # S7–S9: Subsystem analyses
    for df, name, color in [
        (sub_hfd, "S7_Subsystem_HFD_vs_SCD", "#8B0000"),
        (sub_kd,  "S8_Subsystem_KD_vs_SCD",  "#00008B"),
        (sub_wd,  "S9_Subsystem_WD_vs_SCD",  "#006400"),
    ]:
        styled_sheet(df.sort_values("N_significant", ascending=False), name, color)

    # S10–S12: Pathway enrichment
    for df, name, color in [
        (enr_hfd, "S10_Enrichment_HFD_vs_SCD", "#8B0000"),
        (enr_kd,  "S11_Enrichment_KD_vs_SCD",  "#00008B"),
        (enr_wd,  "S12_Enrichment_WD_vs_SCD",  "#006400"),
    ]:
        styled_sheet(df.sort_values("enrichment_ratio", ascending=False), name, color)

print(f"Done. Output: {OUTPUT_FILE}")
print(f"  Sheets: S1-S12 (6 reaction tables + 3 subsystem + 3 enrichment)")