#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import json
import argparse
import traceback
from typing import Dict, Set, Tuple, List, Iterable

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cobra
from cobra.io import load_json_model
from cobra.flux_analysis import flux_variability_analysis
from scipy.stats import ttest_ind, ranksums, mannwhitneyu
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

# =====================================
# Group handling
# =====================================
GROUP_ALIASES = {
    "SCD": {"SCD", "SC", "CD", "CHOW", "ND", "CONTROL", "STDCHOW", "STANDARDCHOW"},
    "WD":  {"WD", "WESTERN", "WESTERNDIET"},
    "HFD": {"HFD", "HF", "HIGHFAT", "HIGH_FAT", "HIGH-FAT"},
    "KD":  {"KD", "KETO", "KETOGENIC", "KETO_DIET", "KETO-DIET"},
    "LFD": {"LFD", "LOWFAT", "LOW_FAT", "LOW-FAT"},
}

def canonical_code(tag: str) -> str:
    t = re.sub(r"[^A-Za-z0-9]+", "", str(tag).upper())
    for canon, aliases in GROUP_ALIASES.items():
        if t == canon or t in aliases:
            return canon
    return t

def parse_groups_from_filename_multi(path: str) -> List[str]:
    base = os.path.basename(path)
    m = re.search(r"_([A-Za-z0-9_]+)_gene_expression\.csv$", base, re.IGNORECASE)
    if not m:
        raise ValueError(f"Filename must contain *_<GROUPS>_gene_expression.csv: {base}")
    chunk = m.group(1)
    raw_tags = [t for t in chunk.split("_") if t]
    tags = [canonical_code(t) for t in raw_tags]
    if len(tags) < 2:
        raise ValueError(f"Need at least 2 group tags in filename; got: {tags}")
    return tags

def choose_default_baseline(tags: List[str]) -> str:
    pref = ["SCD", "WD"]
    for p in pref:
        if p in tags:
            return p
    return tags[0]

# =====================================
# Gene ID mapping
# =====================================
def load_symbol_to_entrez_mapping(mapping_file: str) -> Dict[str, str]:
    """
    Load mapping from gene symbol to entrez ID.
    Returns dict: {lowercase_symbol: entrez_id_as_string}
    """
    df = pd.read_csv(mapping_file)
    mapping = {}
    for _, row in df.iterrows():
        symbol = str(row.get('symbol', '')).strip().lower()
        
        # Handle Entrez ID - convert from float to int to string to remove .0
        entrez_raw = row.get('entrez', '')
        if pd.isna(entrez_raw) or entrez_raw == '':
            continue
        try:
            # Convert to int first (handles floats like 239559.0)
            entrez = str(int(float(entrez_raw)))
        except (ValueError, TypeError):
            continue
            
        if symbol and entrez:
            mapping[symbol] = entrez
            
        # Also add aliases if present
        aliases_str = str(row.get('alias', '')).strip()
        if aliases_str and aliases_str != 'nan':
            for alias in aliases_str.split(','):
                alias = alias.strip().lower()
                if alias and alias not in mapping:
                    mapping[alias] = entrez
    return mapping

# =====================================
# Objective selection
# =====================================
def set_objective_reaction(model, objective_id=None, objective_regex=None, sense="max", logger=print):
    """
    Select objective by id or regex; fallback to biomass/ATPM/first.
    sense: 'max' or 'min'
    """
    chosen = None
    if objective_id:
        try:
            chosen = model.reactions.get_by_id(objective_id)
        except KeyError:
            logger(f"[WARN] Objective id '{objective_id}' not found; will try regex/fallbacks.")
    if (chosen is None) and objective_regex:
        rx = re.compile(objective_regex, re.IGNORECASE)
        for rxn in model.reactions:
            if rx.search(rxn.id) or rx.search(rxn.name or "") or (rxn.subsystem and rx.search(rxn.subsystem)):
                chosen = rxn; break
        if chosen is None:
            logger(f"[WARN] No reaction matched objective_regex '{objective_regex}'.")
    if chosen is None:
        for rxn in model.reactions:
            nm = (rxn.name or "").lower()
            if "biomass" in nm or (rxn.subsystem and "biomass" in rxn.subsystem.lower()):
                chosen = rxn; break
    if chosen is None:
        try:
            chosen = model.reactions.get_by_id("ATPM")
        except KeyError:
            chosen = model.reactions[0]
    # Apply sense
    try:
        model.objective = chosen
        if sense.lower().startswith("min"):
            model.objective_direction = "min"
        else:
            model.objective_direction = "max"
    except Exception:
        pass
    logger(f"[INFO] Objective set to: {chosen.id} ({chosen.name}) | sense={sense}")
    return model, chosen.id

# =====================================
# Reaction classification (EX / transporter / internal)
# =====================================
def classify_reactions(
    model,
    transporter_strategy="e_to_non_e",
    transporter_regex=None,
    transporter_subsystem_regex=None,
    compartments_for_transport=("e",),
) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    transporter_strategy:
      - 'e_to_non_e' (default): any reaction with an 'e' metabolite and any non-'e' metabolite
      - 'regex': reactions whose id matches transporter_regex OR subsystem matches transporter_subsystem_regex
      - 'either': union of e_to_non_e and regex strategies
    """
    ex_rxns, transporter_rxns, internal_rxns = set(), set(), set()

    rx_id_re = re.compile(transporter_regex, re.IGNORECASE) if transporter_regex else None
    rx_subsys_re = re.compile(transporter_subsystem_regex, re.IGNORECASE) if transporter_subsystem_regex else None
    comp_set = set(compartments_for_transport or [])

    def is_transporter_by_e_non_e(rxn) -> bool:
        comps = set([m.compartment for m in rxn.metabolites])
        return (len(comp_set & comps) > 0) and any((c not in comp_set) for c in comps)

    def is_transporter_by_regex(rxn) -> bool:
        if rx_id_re and rx_id_re.search(rxn.id):
            return True
        if rx_subsys_re and rxn.subsystem and rx_subsys_re.search(rxn.subsystem):
            return True
        return False

    for rxn in model.reactions:
        rid = rxn.id
        if rid.startswith("EX_"):
            ex_rxns.add(rid); continue
        by_e_non_e = is_transporter_by_e_non_e(rxn)
        by_regex = is_transporter_by_regex(rxn)
        pick = False
        if transporter_strategy == "e_to_non_e":
            pick = by_e_non_e
        elif transporter_strategy == "regex":
            pick = by_regex
        elif transporter_strategy == "either":
            pick = by_e_non_e or by_regex
        else:
            pick = by_e_non_e
        if pick:
            transporter_rxns.add(rid)
        else:
            internal_rxns.add(rid)
    return ex_rxns, transporter_rxns, internal_rxns

# =====================================
# Expression mapping (E-flux style), scoped - FIXED VERSION
# =====================================
def _standardize_gene_id(gene_id: str) -> str:
    """Standardize gene ID by removing special characters and converting to lowercase"""
    return re.sub(r"[()\-'\",;]", "", str(gene_id).lower().strip())

def _build_gene_index(gene_names):
    """Build index from gene names/IDs to their position in the expression vector"""
    d = {}
    for idx, g in enumerate(gene_names):
        g_str = str(g).strip()
        
        # Store the original
        d[g_str] = idx
        
        # Store standardized version (lowercase, no special chars)
        standardized = _standardize_gene_id(g)
        if standardized != g_str:
            d[standardized] = idx
        
        # Store lowercase version
        g_lower = g_str.lower()
        if g_lower != g_str and g_lower != standardized:
            d[g_lower] = idx
            
        # If it looks like a number, also store without leading zeros and .0
        try:
            g_num = str(int(float(g_str)))
            if g_num != g_str:
                d[g_num] = idx
        except (ValueError, TypeError):
            pass
    
    return d

def _map_expression_to_reactions(model, gene_index, expr_vector, symbol_to_entrez=None):
    """
    Map gene expression to reactions using GPR rules.
    
    Args:
        model: COBRA model
        gene_index: dict mapping gene identifiers to expression vector indices
        expr_vector: numpy array of expression values
        symbol_to_entrez: dict mapping gene symbols to entrez IDs (optional)
    """
    rxn_expr = np.zeros(len(model.reactions))
    matched_genes = set()
    
    for i, rxn in enumerate(model.reactions):
        rule = rxn.gene_reaction_rule
        if not rule:
            continue
        
        rule_clean = re.sub(r"[()]", "", rule)
        or_parts = rule_clean.split(" or ")
        values = []
        
        for part in or_parts:
            and_parts = part.split(" and ")
            vals = []
            
            for gene in and_parts:
                gene = gene.strip()
                if not gene:
                    continue
                
                # Try multiple matching strategies
                expr_value = None
                matched_key = None
                
                # Strategy 1: Direct match with original gene ID
                if gene in gene_index:
                    expr_value = expr_vector[gene_index[gene]]
                    matched_key = gene
                
                # Strategy 2: Standardized match
                if expr_value is None:
                    key = _standardize_gene_id(gene)
                    if key in gene_index:
                        expr_value = expr_vector[gene_index[key]]
                        matched_key = key
                
                # Strategy 3: Lowercase match
                if expr_value is None:
                    key_lower = gene.lower()
                    if key_lower in gene_index:
                        expr_value = expr_vector[gene_index[key_lower]]
                        matched_key = key_lower
                
                # Strategy 4: Try as integer (remove .0 if present)
                if expr_value is None:
                    try:
                        gene_as_int = str(int(float(gene)))
                        if gene_as_int in gene_index:
                            expr_value = expr_vector[gene_index[gene_as_int]]
                            matched_key = gene_as_int
                    except (ValueError, TypeError):
                        pass
                
                if expr_value is not None and expr_value > 0:
                    vals.append(expr_value)
                    if matched_key:
                        matched_genes.add(matched_key)
            
            pos_vals = [v for v in vals if v > 0]
            if pos_vals:
                values.append(min(pos_vals))  # AND = min
        
        if values:
            rxn_expr[i] = max(values)  # OR = max
    
    return rxn_expr, matched_genes

def apply_expression_constraints_scoped(
    model, gene_names, expr_vector, scope_rxn_ids: Set[str],
    *, eflux_quantile=0.95, eflux_floor=0.1, eflux_cap=1000.0, logger=print, label="",
    symbol_to_entrez=None
):
    """
    Apply E-Flux style expression constraints to reactions in a specified scope.
    
    Args:
        symbol_to_entrez: Optional dict mapping gene symbols to Entrez IDs
    """
    # Normalize expression
    pos = expr_vector[expr_vector > 0]
    denom = np.quantile(pos, eflux_quantile) if len(pos) else 1.0
    x = expr_vector / max(denom, 1e-9)
    x = np.clip(x, eflux_floor, eflux_cap)

    # Build gene index
    gene_index = _build_gene_index(gene_names)
    
    # Map expression to reactions
    rxn_expr, matched_genes = _map_expression_to_reactions(
        model, gene_index, x, symbol_to_entrez=symbol_to_entrez
    )

    # Apply constraints
    changed = 0
    changed_rxn_ids = set()
    for rxn, val in zip(model.reactions, rxn_expr):
        if rxn.id not in scope_rxn_ids:
            continue
        if val > 0:
            old_lb, old_ub = rxn.lower_bound, rxn.upper_bound
            if rxn.lower_bound < 0:
                rxn.lower_bound = max(rxn.lower_bound, -float(val))
            rxn.upper_bound = min(rxn.upper_bound, float(val))
            if (rxn.lower_bound != old_lb) or (rxn.upper_bound != old_ub):
                changed += 1
                changed_rxn_ids.add(rxn.id)
    
    logger(f"[INFO] Layer {label}: expression constraints applied to {changed} reactions in scope={len(scope_rxn_ids)}. Genes matched: {len(matched_genes)}")
    return model, changed, len(matched_genes), len(gene_index), matched_genes, changed_rxn_ids

# =====================================
# Layer 1: Diet bounds (with optional unit conversion)
# =====================================
def load_mw_table(path: str) -> Dict[str, float]:
    tbl = pd.read_csv(path)
    out = {}
    for _, row in tbl.iterrows():
        rid = str(row["exchange_id"]).strip()
        mw = float(row["mw_g_per_mol"])
        out[rid] = mw
    return out

def convert_bound_value(lb_model_units: float, *, units: str, ex_id: str, mw_map: Dict[str, float], gDW: float, hours_per_day: float):
    if units == "model":
        return float(lb_model_units)
    if units == "mmol_per_day":
        return float(lb_model_units) / max(gDW, 1e-9) / max(hours_per_day, 1e-9)
    if units == "g_per_day":
        mw = mw_map.get(ex_id)
        if mw is None:
            raise ValueError(f"No MW for {ex_id} in --mw_table; required for g/day conversion.")
        mmol_per_day = (lb_model_units * 1000.0) / mw
        return mmol_per_day / max(gDW, 1e-9) / max(hours_per_day, 1e-9)
    raise ValueError(f"Unsupported units: {units}")

def apply_diet_bounds_layer1(model, *, code: str, diet_bounds: Dict, diet_units: str,
                             mw_map: Dict[str, float] = None, gDW: float = 1.0, hours_per_day: float = 24.0,
                             logger=print):
    code = canonical_code(code)
    if not diet_bounds or code not in diet_bounds:
        logger(f"[INFO] No diet bounds for {code}. Skipping Layer 1.")
        return model, 0
    changed = 0
    for rxn_id, bounds in diet_bounds[code].items():
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
            lb, ub = bounds[0], bounds[1]
        elif isinstance(bounds, dict) and "lb" in bounds and "ub" in bounds:
            lb, ub = bounds["lb"], bounds["ub"]
        else:
            raise ValueError(f"Invalid bounds format for {rxn_id}: {bounds}")
        try:
            rxn = model.reactions.get_by_id(rxn_id)
        except KeyError:
            logger(f"[WARN] Diet lists {rxn_id} but it is not in model. Skipping.")
            continue
        if not rxn.id.startswith("EX_"):
            continue
        new_lb = convert_bound_value(lb, units=diet_units, ex_id=rxn_id, mw_map=(mw_map or {}), gDW=gDW, hours_per_day=hours_per_day)
        new_ub = convert_bound_value(ub, units=diet_units, ex_id=rxn_id, mw_map=(mw_map or {}), gDW=gDW, hours_per_day=hours_per_day) if diet_units != "model" else float(ub)
        old_lb, old_ub = rxn.lower_bound, rxn.upper_bound
        rxn.lower_bound, rxn.upper_bound = float(new_lb), float(new_ub)
        if (rxn.lower_bound != old_lb) or (rxn.upper_bound != old_ub):
            changed += 1
    logger(f"[INFO] Layer 1 (Diet) applied to {changed} EX_ reactions for code={code}.")
    return model, changed

def validate_model(model, logger=print):
    inc = [rxn for rxn in model.reactions if rxn.lower_bound > rxn.upper_bound]
    for rxn in inc:
        mid = (rxn.lower_bound + rxn.upper_bound) / 2
        rxn.lower_bound = mid * 0.9
        rxn.upper_bound = mid * 1.1
    if inc:
        logger(f"[INFO] Fixed {len(inc)} inconsistent bounds in model.")
    return model

# ===================================== 
# Analysis & plotting
# =====================================
def _transform_for_pca(X: np.ndarray, mode: str = "none"):
    mode = (mode or "none").lower()
    Y = X.copy().astype(float)
    def sign_log1p(a):
        return np.sign(a) * np.log1p(np.abs(a))
    if mode in ("log", "log1p", "log-scale", "log_scale"):
        return sign_log1p(Y)
    if mode in ("zscore", "standardize", "standardise", "standardize_features"):
        mu = np.nanmean(Y, axis=0); sd = np.nanstd(Y, axis=0); sd[sd == 0] = 1.0
        return (Y - mu) / sd
    if mode in ("log_zscore", "log1p_zscore", "log_then_zscore"):
        Y = sign_log1p(Y)
        mu = np.nanmean(Y, axis=0); sd = np.nanstd(Y, axis=0); sd[sd == 0] = 1.0
        return (Y - mu) / sd
    return Y

def analyze_flux_distributions(model, solutions_dict, baseline_key, results_dir="results", pca_scale="none", write_replicates_long=False, logger=print, replicate_labels=None):
    os.makedirs(results_dir, exist_ok=True)
    out_dir = os.path.join(results_dir, "flux_analysis")
    os.makedirs(out_dir, exist_ok=True)
    cond_names = list(solutions_dict.keys())
    flux_data = {cond: np.array([sol.fluxes.values for sol in sols]) for cond, sols in solutions_dict.items()}
    rep_name_map = {}
    for cond in cond_names:
        n = flux_data[cond].shape[0]
        if replicate_labels and cond in replicate_labels and len(replicate_labels[cond]) == n:
            rep_name_map[cond] = list(replicate_labels[cond])
        else:
            rep_name_map[cond] = [f"{cond}_rep{i+1}" for i in range(n)]
    cols = ["ReactionID", "ReactionName", "Subsystem"]
    for cond in cond_names:
        cols += [f"{cond}_MeanFlux", f"{cond}_StdFlux", f"{cond}_N"]
        for rn in rep_name_map[cond]:
            cols.append(f"{rn}_Flux")
    for c in [x for x in cond_names if x != baseline_key]:
        cols += [f"Diff({c}-{baseline_key})", f"Ratio({c}/{baseline_key})"]
    df = pd.DataFrame(columns=cols)
    for rxn in model.reactions:
        i = model.reactions.index(rxn)
        row = {"ReactionID": rxn.id, "ReactionName": rxn.name, "Subsystem": rxn.subsystem or ""}
        for cond in cond_names:
            arr = flux_data[cond][:, i]
            row[f"{cond}_MeanFlux"] = float(np.mean(arr)) if arr.size else float('nan')
            row[f"{cond}_StdFlux"]  = float(np.std(arr)) if arr.size else float('nan')
            row[f"{cond}_N"]        = int(arr.shape[0])
            for r, v in enumerate(arr, start=0):
                row[f"{rep_name_map[cond][r]}_Flux"] = float(v)
        ref_vals = flux_data[baseline_key][:, i]
        ref_mean = float(np.mean(ref_vals)) if ref_vals.size else float('nan')
        ref_mean_abs = float(np.mean(np.abs(ref_vals)) + 1e-12) if ref_vals.size else float('nan')
        for c in [x for x in cond_names if x != baseline_key]:
            c_vals = flux_data[c][:, i]
            c_mean = float(np.mean(c_vals)) if c_vals.size else float('nan')
            row[f"Diff({c}-{baseline_key})"] = c_mean - ref_mean if (not np.isnan(c_mean) and not np.isnan(ref_mean)) else float('nan')
            row[f"Ratio({c}/{baseline_key})"] = (float(np.mean(np.abs(c_vals))) / ref_mean_abs) if (c_vals.size and not np.isnan(ref_mean_abs) and ref_mean_abs != 0.0) else float('nan')
        df.loc[len(df)] = row
    out_csv = os.path.join(out_dir, "reaction_flux_comparison_extended.csv")
    df.to_csv(out_csv, index=False); logger(f"[INFO] Extended flux comparison saved to: {out_csv}")
    if write_replicates_long:
        long_records = []
        for cond, arr in flux_data.items():
            for r in range(arr.shape[0]):
                rep_label = rep_name_map[cond][r]
                for i, rxn in enumerate(model.reactions):
                    long_records.append({
                        "ReactionID": rxn.id, "ReactionName": rxn.name, "Subsystem": rxn.subsystem or "",
                        "Group": cond, "Replicate": r + 1, "ReplicateName": rep_label, "Flux": float(arr[r, i]),
                    })
        long_df = pd.DataFrame(long_records)
        long_csv = os.path.join(out_dir, "flux_replicates_long.csv")
        long_df.to_csv(long_csv, index=False); logger(f"[INFO] Replicate-level long table saved to: {long_csv}")
    all_vectors, all_labels = [], []
    for cond, arr in flux_data.items():
        for r in range(arr.shape[0]):
            all_vectors.append(arr[r, :])
            label = rep_name_map[cond][r] if arr.shape[0] > 0 else cond
            all_labels.append(label)
    all_vectors = np.array(all_vectors)
    if all_vectors.shape[0] >= 2:
        Xp_in = _transform_for_pca(all_vectors, mode=pca_scale)
        pca = PCA(n_components=2)
        Xp = pca.fit_transform(Xp_in)
        plt.figure(figsize=(7, 6))
        plt.scatter(Xp[:, 0], Xp[:, 1], s=50, alpha=0.8)
        for idx, label in enumerate(all_labels):
            plt.text(Xp[idx, 0], Xp[idx, 1], label, fontsize=8)
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        plt.title("PCA of Flux Distributions")
        plt.grid(True)
        pca_fig = os.path.join(out_dir, "flux_pca.png")
        plt.savefig(pca_fig); plt.close()
        logger(f"[INFO] PCA plot saved to: {pca_fig}")

# =====================================
# Stats with BH-FDR
# =====================================
def bh_fdr(pvals: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(pvals), dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranks = np.arange(1, m+1, dtype=float)
    q = np.empty_like(p)
    q[order] = p[order] * (m / ranks)
    # monotone
    for i in range(m-2, -1, -1):
        q[order[i]] = min(q[order[i]], q[order[i+1]])
    q = np.clip(q, 0.0, 1.0)
    return q

def compare_flux_statistically(solutions_dict, model, results_dir="results", test_type="t-test", logger=print):
    out_dir = os.path.join(results_dir, "stats_comparison")
    os.makedirs(out_dir, exist_ok=True)
    cond_names = list(solutions_dict.keys())
    flux_data = {cond: np.array([sol.fluxes.values for sol in sols]) for cond, sols in solutions_dict.items()}
    records = []
    for i, rxn in enumerate(model.reactions):
        for cA in cond_names:
            for cB in cond_names:
                if cA >= cB: continue
                arrA = flux_data[cA][:, i]
                arrB = flux_data[cB][:, i]
                if len(arrA) < 2 or len(arrB) < 2:
                    pval = float('nan'); stat = float('nan')
                else:
                    if test_type.lower() == "wilcoxon":
                        stat, pval = ranksums(arrA, arrB)
                    elif test_type.lower() == "mann-whitney":
                        stat, pval = mannwhitneyu(arrA, arrB, alternative='two-sided')
                    else:
                        stat, pval = ttest_ind(arrA, arrB, equal_var=False)
                mA = float(np.mean(arrA)) if arrA.size else float('nan')
                mB = float(np.mean(arrB)) if arrB.size else float('nan')
                records.append({
                    "ReactionID": rxn.id, "ReactionName": rxn.name, "Subsystem": rxn.subsystem or "",
                    "GroupA": cA, "GroupB": cB,
                    "MeanA": mA, "MeanB": mB,
                    "Diff(A-B)": mA - mB if (not np.isnan(mA) and not np.isnan(mB)) else float('nan'),
                    "TestStat": float(stat) if not np.isnan(stat) else float('nan'),
                    "PValue": float(pval) if not np.isnan(pval) else float('nan')
                })
    df = pd.DataFrame(records)
    finite_p = df["PValue"].dropna().values
    if len(finite_p) > 0:
        q = bh_fdr(finite_p)
        df.loc[df["PValue"].notna(), "FDR_BH"] = q
    else:
        df["FDR_BH"] = float('nan')
    out_csv = os.path.join(out_dir, "flux_pairwise_stats.csv")
    df.to_csv(out_csv, index=False)
    logger(f"[INFO] Pairwise stats + BH-FDR saved to: {out_csv}")
    # Global distance matrix (Euclidean over flux vectors)
    rep_labels = globals().get('_GLOBAL_REP_NAME_MAP_FOR_STATS', None)
    all_vectors, all_names = [], []
    for cond in cond_names:
        arr = flux_data[cond]
        if rep_labels and cond in rep_labels:
            names_list = rep_labels[cond]
        else:
            names_list = [f"{cond}_rep{r+1}" for r in range(arr.shape[0])]
        for r in range(arr.shape[0]):
            all_vectors.append(arr[r, :])
            all_names.append(names_list[r])
    all_vectors = np.array(all_vectors)
    if all_vectors.shape[0] > 1:
        dmat = squareform(pdist(all_vectors, metric='euclidean'))
        dmat_df = pd.DataFrame(dmat, index=all_names, columns=all_names)
        dmat_csv = os.path.join(out_dir, "flux_global_distance_matrix.csv")
        dmat_df.to_csv(dmat_csv)
        logger(f"[INFO] Global distance matrix saved to: {dmat_csv}")
    return df

# =====================================
# Rank-Product
# =====================================
def compute_rank_product(flux_csv: str, baseline: str, targets: List[str], out_csv: str, logger=print):
    df = pd.read_csv(flux_csv)
    df = df.set_index("ReactionID")
    diff_cols = [c for c in df.columns if c.startswith("Diff(")]
    valid_diff_cols = [c for c in diff_cols if any(t in c for t in targets)]
    if not valid_diff_cols:
        logger(f"[WARN] No diff columns found for targets={targets} in flux_csv.")
        return pd.DataFrame()
    sub_df = df[valid_diff_cols].copy()
    # Rank each diff col
    ranks = {}
    for col in sub_df.columns:
        # s = sub_df[col].dropna() 
        s = sub_df[col].dropna().abs()                      # If your goal was to find the most responsive reactions in either direction (up or down)
        ranked = s.rank(method="average", ascending=False)  #  you might need a second rank product run for downregulated reactions using ascending=True
        ranks[col] = ranked
    rank_df = pd.DataFrame(ranks)
    # Geometric mean of ranks
    rp = rank_df.apply(lambda row: np.exp(np.log(row + 1e-9).mean()), axis=1)
    rp_sorted = rp.sort_values(ascending=True)
    rp_df = pd.DataFrame({
        "ReactionID": rp_sorted.index,
        "RankProduct": rp_sorted.values,
        "RankProductRank": range(1, len(rp_sorted)+1)
    })
    for col in valid_diff_cols:
        rp_df[col] = df.loc[rp_sorted.index, col].values
    rp_df.to_csv(out_csv, index=False)
    logger(f"[INFO] Rank-product saved to: {out_csv}")
    return rp_df

# =====================================
# Extended visuals
# =====================================
def extended_visuals(model, solutions_dict, baseline_key, results_dir="results", logger=print):
    out_dir = os.path.join(results_dir, "extended_visuals")
    os.makedirs(out_dir, exist_ok=True)
    cond_names = list(solutions_dict.keys())
    flux_data = {cond: np.array([sol.fluxes.values for sol in sols]) for cond, sols in solutions_dict.items()}
    # Heatmap
    mean_fluxes = []
    for cond in cond_names:
        arr = flux_data[cond]
        mean_fluxes.append(np.mean(arr, axis=0))
    heatmap_data = np.array(mean_fluxes).T
    fig, ax = plt.subplots(figsize=(8, 10))
    cax = ax.imshow(heatmap_data, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_xlabel("Condition")
    ax.set_ylabel("Reaction Index")
    ax.set_title("Mean Flux Heatmap")
    ax.set_xticks(range(len(cond_names)))
    ax.set_xticklabels(cond_names)
    fig.colorbar(cax, ax=ax, label="Flux")
    out_fig = os.path.join(out_dir, "mean_flux_heatmap.png")
    plt.savefig(out_fig, dpi=150, bbox_inches='tight'); plt.close()
    logger(f"[INFO] Mean flux heatmap saved to: {out_fig}")

# =====================================
# Cytoscape edges
# =====================================
def build_cytoscape_edges_for_comparison(model, diff_series, ratio_series, cond, baseline, abs_diff_threshold=0.0, out_path=None, logger=print):
    edges = []
    for rxn in model.reactions:
        rid = rxn.id
        if rid not in diff_series.index:
            continue
        diff_val = diff_series.loc[rid]
        if np.isnan(diff_val) or abs(diff_val) < abs_diff_threshold:
            continue
        ratio_val = ratio_series.loc[rid] if (rid in ratio_series.index) else float('nan')
        # Build edges from reaction metabolites
        for met, coeff in rxn.metabolites.items():
            mid = met.id
            edges.append({
                "ReactionID": rid,
                "ReactionName": rxn.name or "",
                "Subsystem": rxn.subsystem or "",
                "MetaboliteID": mid,
                "MetaboliteName": met.name or "",
                "Coefficient": float(coeff),
                f"Diff({cond}-{baseline})": float(diff_val),
                f"Ratio({cond}/{baseline})": float(ratio_val)
            })
    df = pd.DataFrame(edges)
    if out_path:
        df.to_csv(out_path, index=False)
        logger(f"[INFO] Cytoscape edges ({len(df)} rows) saved to: {out_path}")
    return df

# =====================================
# Report files
# =====================================
def write_report_files(report: dict, results_dir: str):
    report_json = os.path.join(results_dir, "analysis_report.json")
    with open(report_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[INFO] Analysis report written to: {report_json}")

# =====================================
# Main pipeline
# =====================================
def run_pipeline(
    rnaseq_file,
    met_data_file=None,
    model_file="iMM1415.json",
    results_dir="results",
    aggregate=False,
    infer_groups_from_filename=True,
    baseline_code=None,
    explicit_groups=None,
    mapping_file=None,  # NEW: symbol-to-entrez mapping file
    column_regex=None,
    eflux_quantile=0.95,
    eflux_floor=0.1,
    eflux_cap=1000.0,
    diet_bounds_json=None,
    diet_bounds_units="model",
    mw_table=None,
    gDW=1.0,
    hours_per_day=24.0,
    pca_scale="none",
    write_replicates_long=False,
    test_type="t-test",
    objective_id=None,
    objective_regex=None,
    objective_sense="max",
    transporter_strategy="e_to_non_e",
    transporter_regex=None,
    transporter_subsystem_regex=None,
    transporter_compartments="e",
    edge_abs_diff_threshold=0.0,
    rank_product_for=None,
    no_fva=False,
    solver="gurobi"
):
    # Load symbol-to-entrez mapping if provided
    symbol_to_entrez = None
    if mapping_file and os.path.exists(mapping_file):
        print(f"[INFO] Loading gene symbol-to-entrez mapping from: {mapping_file}")
        symbol_to_entrez = load_symbol_to_entrez_mapping(mapping_file)
        print(f"[INFO] Loaded {len(symbol_to_entrez)} symbol->entrez mappings")
    
    # Groups
    if infer_groups_from_filename:
        tags = parse_groups_from_filename_multi(rnaseq_file)
    elif explicit_groups:
        tags = [canonical_code(x) for x in re.split(r"[,\s]+", explicit_groups) if x]
    else:
        raise ValueError("Must set --infer_groups_from_filename OR --explicit_groups.")
    baseline = canonical_code(baseline_code) if baseline_code else choose_default_baseline(tags)
    if baseline not in tags:
        baseline = tags[0]
    print(f"[INFO] Detected groups: {tags} (baseline={baseline})")

    # Report
    report = {
        "groups": {"all_groups": tags, "baseline": baseline, "sample_counts": {}},
        "rna_seq": {"file": rnaseq_file, "total_genes": 0, "genes_with_expr": 0},
        "parameters": {
            "aggregate": aggregate,
            "eflux_quantile": eflux_quantile,
            "eflux_floor": eflux_floor,
            "eflux_cap": eflux_cap,
            "diet_bounds_json": diet_bounds_json,
            "diet_bounds_units": diet_bounds_units,
            "mw_table": mw_table,
            "gDW": gDW,
            "hours_per_day": hours_per_day,
            "test_type": test_type,
            "objective_id": objective_id,
            "objective_regex": objective_regex,
            "objective_sense": objective_sense,
            "transporter_strategy": transporter_strategy,
            "transporter_regex": transporter_regex,
            "transporter_subsystem_regex": transporter_subsystem_regex,
            "no_fva": no_fva
        },
        "per_group": {},
        "global": {},
        "outputs": {}
    }
    os.makedirs(results_dir, exist_ok=True)

    # Configure solver FIRST (global configuration)
    # This must be done BEFORE loading the model so the model inherits the solver
    print(f"[INFO] Configuring global solver: {solver}")
    try:
        cobra.Configuration().solver = solver
        print(f"[INFO] Global solver successfully configured: {solver}")
    except Exception as e:
        print(f"[WARNING] Could not set solver to '{solver}': {e}")
        print(f"[INFO] Will use default solver instead")
    
    # Model (now inherits the global solver configuration)
    print("[STEP] Loading model...")
    model = load_json_model(model_file)
    print(f"[INFO] Model loaded with solver: {model.solver.interface.__name__}")
    
    model, obj_id = set_objective_reaction(model, objective_id=objective_id, objective_regex=objective_regex, sense=objective_sense)
    report["model"] = {
        "num_reactions": len(model.reactions),
        "num_metabolites": len(model.metabolites),
        "num_genes": len(model.genes),
        "objective_id": obj_id,
        "objective_name": getattr(getattr(model, "objective", None), "name", "unknown"),
        "solver": str(model.solver.interface.__name__)
    }

    # RNA-seq
    rna = pd.read_csv(rnaseq_file)
    gene_symbols = rna["Gene_Symbol"].astype(str).str.lower().tolist()
    
    # NEW: Convert symbols to Entrez IDs if mapping is available
    gene_ids_for_matching = gene_symbols  # Default to symbols
    if symbol_to_entrez:
        gene_ids_for_matching = []
        conversion_stats = {"matched": 0, "unmatched": 0}
        for sym in gene_symbols:
            sym_lower = sym.lower().strip()
            if sym_lower in symbol_to_entrez:
                gene_ids_for_matching.append(symbol_to_entrez[sym_lower])
                conversion_stats["matched"] += 1
            else:
                gene_ids_for_matching.append(sym)  # Keep original if no mapping
                conversion_stats["unmatched"] += 1
        print(f"[INFO] Gene ID conversion: {conversion_stats['matched']} matched, {conversion_stats['unmatched']} unmatched")
        
        # Debug: Show sample conversions
        print(f"[DEBUG] Sample gene conversions:")
        for i in range(min(10, len(gene_symbols))):
            if gene_symbols[i].lower() in symbol_to_entrez:
                print(f"   {gene_symbols[i]} -> {gene_ids_for_matching[i]}")
        
        # Debug: Check if converted IDs exist in model
        with open(model_file, 'r') as f:
            model_json = json.load(f)
        model_gene_ids = set([str(g['id']) for g in model_json.get('genes', [])])
        converted_in_model = sum(1 for gid in gene_ids_for_matching if gid in model_gene_ids)
        print(f"[DEBUG] Converted genes that exist in model: {converted_in_model}/{conversion_stats['matched']}")
    
    sample_cols = rna.columns[2:].tolist()
    expr = rna.iloc[:, 2:].to_numpy()
    report["rna_seq"]["total_genes"] = int(len(gene_symbols))
    report["rna_seq"]["genes_with_expr"] = int((expr.sum(axis=1) > 0).sum())
    def cols_for(tag):
        if column_regex:
            pat = column_regex
            pat = pat.replace("{TAG}", re.escape(tag))
            aliases = GROUP_ALIASES.get(tag, {tag}) | {tag}
            alias_alt = "(?:" + "|".join(sorted(re.escape(a) for a in aliases)) + ")"
            pat = pat.replace("{ALIASES}", alias_alt)
        else:
            aliases = GROUP_ALIASES.get(tag, {tag}) | {tag}
            alias_alt = "(?:" + "|".join(sorted(re.escape(a) for a in aliases)) + ")"
            pat = rf"(?<![A-Za-z0-9]){alias_alt}(?![A-Za-z0-9])"
        rx = re.compile(pat, re.IGNORECASE)
        return [c for c in sample_cols if rx.search(c)]
    tag_to_indices = {tag: [sample_cols.index(c) for c in cols_for(tag)] for tag in tags}
    for _tg, _idx in tag_to_indices.items():
        report["groups"]["sample_counts"][_tg] = int(len(_idx))
    for tag, idxs in tag_to_indices.items():
        if not idxs:
            raise ValueError(f"No columns found for group '{tag}' in RNA-seq file.")
        print(f"[DEBUG] {tag}: matched {len(idxs)} RNA-seq columns -> {[sample_cols[i] for i in idxs]}")

    # Diet bounds
    diet_bounds = None
    if diet_bounds_json:
        with open(diet_bounds_json, "r") as f:
            raw = json.load(f)
        diet_bounds = {canonical_code(k): {rid: [float(v[0]), float(v[1])] for rid, v in d.items()} for k, d in raw.items()}
    if diet_bounds_units not in {"model","g_per_day","mmol_per_day"}:
        raise ValueError("--diet_bounds_units must be one of: model, g_per_day, mmol_per_day")
    mw_map = load_mw_table(mw_table) if (mw_table and diet_bounds_units=="g_per_day") else {}

    # Classify reactions (custom transporter logic)
    comps = tuple([c.strip() for c in transporter_compartments.split(",") if c.strip()]) if isinstance(transporter_compartments, str) else tuple(transporter_compartments)
    ex_set, trans_set, int_set = classify_reactions(
        model,
        transporter_strategy=transporter_strategy,
        transporter_regex=transporter_regex,
        transporter_subsystem_regex=transporter_subsystem_regex,
        compartments_for_transport=comps
    )
    print(f"[INFO] Reaction classes: EX={len(ex_set)} | Transporters={len(trans_set)} | Internal={len(int_set)}")

    # Solve per group (replicate-wise or aggregate)
    solutions_dict = {}
    fva_queue = []
    for tag in tags:
        idxs = tag_to_indices[tag]
        if aggregate:
            vec = np.mean(expr[:, idxs], axis=1)
            mdl = model.copy()
            mdl, L1_count = apply_diet_bounds_layer1(mdl, code=tag, diet_bounds=diet_bounds, diet_units=diet_bounds_units,
                                                     mw_map=mw_map, gDW=gDW, hours_per_day=hours_per_day)
            mdl, L2_changed, genes_matched, gene_total, genes_set, rxn_set_L2 = apply_expression_constraints_scoped(
                mdl, gene_ids_for_matching, vec, trans_set,
                eflux_quantile=eflux_quantile, eflux_floor=eflux_floor, eflux_cap=eflux_cap, label="L2",
                symbol_to_entrez=symbol_to_entrez)
            mdl, L3_changed, genes_matched3, gene_total3, genes_set3, rxn_set_L3 = apply_expression_constraints_scoped(
                mdl, gene_ids_for_matching, vec, int_set,
                eflux_quantile=eflux_quantile, eflux_floor=eflux_floor, eflux_cap=eflux_cap, label="L3",
                symbol_to_entrez=symbol_to_entrez)
            mdl = validate_model(mdl)
            sol = mdl.optimize()
            print(f"[{tag}] objective={sol.objective_value:.6f} | L1={L1_count} | L2_rxns={L2_changed} | L3_rxns={L3_changed}")
            solutions_dict[tag] = [sol]
            fva_queue.append((tag, mdl, sol))
            report["per_group"][tag] = {
                "samples": int(len(idxs)),
                "objective_values": [float(sol.objective_value)],
                "objective_mean": float(sol.objective_value),
                "objective_std": 0.0,
                "rxns_constrained_unique_L2": int(len(set(rxn_set_L2))),
                "rxns_constrained_unique_L3": int(len(set(rxn_set_L3))),
                "genes_mapped_unique": int(len(set(genes_set) | set(genes_set3)))
            }
        else:
            sols, obj_vals = [], []
            rxn_union_L2, rxn_union_L3 = set(), set()
            gene_union = set()
            for i, si in enumerate(idxs):
                vec = expr[:, si]
                mdl = model.copy()
                mdl, L1_count = apply_diet_bounds_layer1(mdl, code=tag, diet_bounds=diet_bounds, diet_units=diet_bounds_units,
                                                         mw_map=mw_map, gDW=gDW, hours_per_day=hours_per_day)
                mdl, L2_changed, genes_matched, gene_total, genes_set, rxn_set_L2 = apply_expression_constraints_scoped(
                    mdl, gene_ids_for_matching, vec, trans_set,
                    eflux_quantile=eflux_quantile, eflux_floor=eflux_floor, eflux_cap=eflux_cap, label="L2",
                    symbol_to_entrez=symbol_to_entrez)
                mdl, L3_changed, genes_matched3, gene_total3, genes_set3, rxn_set_L3 = apply_expression_constraints_scoped(
                    mdl, gene_ids_for_matching, vec, int_set,
                    eflux_quantile=eflux_quantile, eflux_floor=eflux_floor, eflux_cap=eflux_cap, label="L3",
                    symbol_to_entrez=symbol_to_entrez)
                mdl = validate_model(mdl)
                sol = mdl.optimize()
                print(f"[{tag}] rep{i+1} objective={sol.objective_value:.6f} | L1={L1_count} | L2_rxns={L2_changed} | L3_rxns={L3_changed}")
                obj_vals.append(float(sol.objective_value))
                rxn_union_L2 |= set(rxn_set_L2); rxn_union_L3 |= set(rxn_set_L3)
                gene_union |= set(genes_set) | set(genes_set3)
                sols.append(sol); fva_queue.append((f"{tag}_rep{i+1}", mdl, sol))
            solutions_dict[tag] = sols
            report["per_group"][tag] = {
                "samples": int(len(idxs)),
                "objective_values": obj_vals,
                "objective_mean": float(np.mean(obj_vals)) if obj_vals else None,
                "objective_std": float(np.std(obj_vals)) if obj_vals else None,
                "rxns_constrained_unique_L2": int(len(set(rxn_union_L2))),
                "rxns_constrained_unique_L3": int(len(set(rxn_union_L3))),
                "genes_mapped_unique": int(len(set(gene_union)))
            }

    # Rep labels -> used in visuals and distance
    replicate_labels = {}
    for tag in tags:
        idxs = tag_to_indices[tag]
        if aggregate:
            replicate_labels[tag] = [tag]
        else:
            replicate_labels[tag] = [sample_cols[i] for i in idxs]
    globals()['_GLOBAL_REP_NAME_MAP_FOR_STATS'] = replicate_labels

    # Flux tables & PCA
    analyze_flux_distributions(model, solutions_dict, baseline_key=baseline, results_dir=results_dir, pca_scale=pca_scale, write_replicates_long=write_replicates_long, replicate_labels=replicate_labels)
    flux_table_csv = os.path.join(results_dir, "flux_analysis", "reaction_flux_comparison_extended.csv")
    report["outputs"]["flux_comparison_csv"] = flux_table_csv
    report["outputs"]["pca_png"] = os.path.join(results_dir, "flux_analysis", "flux_pca.png")

    # FVA
    if not no_fva:
        out_dir = os.path.join(results_dir, "fva")
        os.makedirs(out_dir, exist_ok=True)
        for name, mdl, sol in fva_queue:
            fva_res = flux_variability_analysis(mdl, fraction_of_optimum=0.95)
            fva_res["range"] = fva_res["maximum"] - fva_res["minimum"]
            with open(os.path.join(out_dir, f"FVA_{name}.txt"), "w") as f:
                f.write(f"# FVA Results for {name}\nRxn\tMin\tMax\tRange\n")
                for rxn, row in fva_res.iterrows():
                    f.write(f"{rxn}\t{row['minimum']:.6f}\t{row['maximum']:.6f}\t{row['range']:.6f}\n")
        print(f"[INFO] FVA results saved in: {out_dir}")
        report["outputs"]["fva_dir"] = out_dir
    else:
        print("[INFO] FVA analysis skipped (--no_fva flag set)")

    # Extended visuals
    extended_visuals(model, solutions_dict, baseline_key=baseline, results_dir=results_dir)

    # Stats + BH-FDR
    stats_df = compare_flux_statistically(solutions_dict, model, results_dir=results_dir, test_type=test_type)
    report["outputs"]["pairwise_stats_csv"] = os.path.join(results_dir, "stats_comparison", "flux_pairwise_stats.csv")
    report["outputs"]["distance_matrix_csv"] = os.path.join(results_dir, "stats_comparison", "flux_global_distance_matrix.csv")

    # Rank-product
    if rank_product_for:
        targets = [canonical_code(x) for x in re.split(r"[,\s]+", rank_product_for) if x]
        rp_csv = os.path.join(results_dir, "rank_product.csv")
        rp_df = compute_rank_product(flux_table_csv, baseline, targets, rp_csv)
        report["outputs"]["rank_product_csv"] = rp_csv
        print(f"[INFO] Rank-product computed for targets={targets}: {rp_csv} (top5):")
        print(rp_df.head(5))

    # Cytoscape edges
    try:
        ft = pd.read_csv(flux_table_csv)
        edge_dir = os.path.join(results_dir, "cytoscape_edges")
        os.makedirs(edge_dir, exist_ok=True)
        ft = ft.set_index("ReactionID")
        for cond in [t for t in tags if t != baseline]:
            diff_col = f"Diff({cond}-{baseline})"
            ratio_col = f"Ratio({cond}/{baseline})"
            if diff_col not in ft.columns:
                continue
            diff_series = ft[diff_col]
            ratio_series = ft[ratio_col] if ratio_col in ft.columns else pd.Series(index=ft.index, data=np.nan)
            out_path = os.path.join(edge_dir, f"edges_{cond}_vs_{baseline}.csv")
            _ = build_cytoscape_edges_for_comparison(
                model, diff_series, ratio_series, cond, baseline,
                abs_diff_threshold=float(edge_abs_diff_threshold), out_path=out_path
            )
        report["outputs"]["cytoscape_edge_dir"] = edge_dir
        print(f"[INFO] Cytoscape edges written to: {edge_dir}")
    except Exception as e:
        print("[WARN] Cytoscape edge build failed:", e)

    # Report files
    write_report_files(report, results_dir)
    print("[INFO] Layered multi-group analysis complete.")

# =====================================
# CLI
# =====================================
def main():
    p = argparse.ArgumentParser(description="Multi-group metabolic analysis with 3-layer constraints (Diet/EX, Transporters, Internal) + BH-FDR, Rank-Product, Cytoscape edges.")
    p.add_argument("rnaseq_file", help="RNA-seq CSV path (e.g., ..._SCD_HFD_KD_gene_expression.csv)")
    p.add_argument("met_data_file", nargs="?", default=None, help="(Optional) metabolite CSV (currently unused)")
    p.add_argument("--model_file", default="iMM1415.json")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--aggregate", action="store_true", help="Aggregate replicates within each group")
    p.add_argument("--pca_scale", default="none", choices=["none","zscore","log","log_zscore","log1p","log1p_zscore"], help="Transform before PCA")
    p.add_argument("--write_replicates_long", action="store_true", help="Write long-format replicate flux table")
    p.add_argument("--infer_groups_from_filename", action="store_true", default=True)
    p.add_argument("--baseline_code", default=None, help="Baseline/control code (e.g., SCD). Default: SCD>WD>first")
    p.add_argument("--explicit_groups", default=None, help="Comma/space-separated list of groups when not inferring from filename")
    p.add_argument("--mapping_file", default=None, help="CSV with columns: entrez,symbol (for GPR conversion)")
    p.add_argument("--column_regex", default=None, help="Override group column regex; use {TAG}/{ALIASES} placeholders")
    p.add_argument("--eflux_quantile", type=float, default=0.95, help="Quantile for per-sample normalization (0<q<=1)")
    p.add_argument("--eflux_floor", type=float, default=0.1, help="Lower floor on normalized expression")
    p.add_argument("--eflux_cap", type=float, default=1000.0, help="Upper cap on normalized expression")
    p.add_argument("--diet_bounds_json", default=None, help="JSON mapping diet-> {rxn_id: [lb, ub], ...}")
    p.add_argument("--diet_bounds_units", default="model", choices=["model", "g_per_day", "mmol_per_day"], help="Units of bounds in the JSON")
    p.add_argument("--mw_table", default=None, help="CSV with columns: exchange_id,mw_g_per_mol (required if --diet_bounds_units g_per_day)")
    p.add_argument("--gDW", type=float, default=1.0, help="Biomass dry weight for unit conversion (gDW)")
    p.add_argument("--hours_per_day", type=float, default=24.0, help="Hours per day for unit conversion")
    p.add_argument("--test_type", default="t-test", choices=["t-test", "wilcoxon", "mann-whitney"])
    # Objective
    p.add_argument("--objective_id", default=None, help="Objective reaction id (exact)")
    p.add_argument("--objective_regex", default=None, help="Objective selection by regex over id/name/subsystem")
    p.add_argument("--objective_sense", default="max", choices=["max","min"], help="Optimize for max or min")
    # Transporters
    p.add_argument("--transporter_strategy", default="e_to_non_e", choices=["e_to_non_e","regex","either"], help="How to detect transporter reactions")
    p.add_argument("--transporter_regex", default=None, help="Regex on reaction id for transporter detection (strategy 'regex'/'either')")
    p.add_argument("--transporter_subsystem_regex", default=None, help="Regex on subsystem for transporter detection")
    p.add_argument("--transporter_compartments", default="e", help="Comma-separated compartments considered 'external-like' (default: e)")
    # Cytoscape
    p.add_argument("--edge_abs_diff_threshold", type=float, default=0.0, help="|Diff| threshold for including reaction edges")
    # Rank-product
    p.add_argument("--rank_product_for", default=None, help="Comma-separated list of target groups (vs baseline) to include in rank-product")
    # FVA control
    p.add_argument("--no_fva", action="store_true", help="Skip FVA (Flux Variability Analysis)")
    # Solver selection
    p.add_argument("--solver", default="gurobi", 
                   help="Optimization solver to use (default: gurobi). Options: gurobi, glpk, cplex, etc.")
    args = p.parse_args()
    try:
        run_pipeline(
            rnaseq_file=args.rnaseq_file,
            met_data_file=args.met_data_file,
            model_file=args.model_file,
            results_dir=args.results_dir,
            aggregate=args.aggregate,
            infer_groups_from_filename=args.infer_groups_from_filename,
            baseline_code=args.baseline_code,
            explicit_groups=args.explicit_groups,
            mapping_file=args.mapping_file,
            column_regex=args.column_regex,
            eflux_quantile=args.eflux_quantile,
            eflux_floor=args.eflux_floor,
            eflux_cap=args.eflux_cap,
            diet_bounds_json=args.diet_bounds_json,
            diet_bounds_units=args.diet_bounds_units,
            mw_table=args.mw_table,
            gDW=args.gDW,
            hours_per_day=args.hours_per_day,
            pca_scale=args.pca_scale,
            write_replicates_long=args.write_replicates_long,
            test_type=args.test_type,
            objective_id=args.objective_id,
            objective_regex=args.objective_regex,
            objective_sense=args.objective_sense,
            transporter_strategy=args.transporter_strategy,
            transporter_regex=args.transporter_regex,
            transporter_subsystem_regex=args.transporter_subsystem_regex,
            transporter_compartments=args.transporter_compartments,
            edge_abs_diff_threshold=args.edge_abs_diff_threshold,
            rank_product_for=args.rank_product_for,
            no_fva=args.no_fva,
            solver=args.solver
        )
    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    sys.exit(main())
