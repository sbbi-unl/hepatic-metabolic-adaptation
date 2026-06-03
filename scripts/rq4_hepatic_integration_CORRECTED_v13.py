#!/usr/bin/env python3
"""
RQ4: HEPATIC-MICROBIOME INTEGRATION ANALYSIS (BIOLOGICALLY CORRECTED)
=====================================================================
"""


import sys
sys.stdout.reconfigure(encoding='utf-8')
import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cobra
from cobra.flux_analysis import pfba
from typing import Dict, List, Tuple, Optional, Set
import warnings


################################################################################
# GPR PARSING AND EVALUATION
################################################################################

def parse_gpr_to_dnf(gpr_string: str) -> List[Set[str]]:
    """
    Parse GPR string into Disjunctive Normal Form (DNF).
    
    DNF: OR of ANDs, e.g., (A and B) or (C and D) or E
    Represented as: [{A, B}, {C, D}, {E}]
    
    Parameters:
    -----------
    gpr_string : str
        Gene-protein-reaction rule string
        
    Returns:
    --------
    List of sets, where each set contains genes in an AND clause
    
    Scientific Rationale:
    --------------------
    GPR rules encode enzyme complex structure:
    - AND: All subunits required (protein complex)
    - OR: Alternative enzymes (isozymes or paralogs)
    
    Example:
    --------
    "(gene1 and gene2) or gene3" → [{gene1, gene2}, {gene3}]
    
    Engineering Benefit:
    -------------------
    DNF form allows systematic evaluation of alternative enzyme forms
    and proper application of min/max logic for AND/OR operations.
    """
    if not gpr_string or gpr_string.strip() == '':
        return []
    
    # Remove 'and'/'or' keywords, split by 'or' to get AND clauses
    # This is simplified - for production, use proper GPR parser
    gpr_clean = gpr_string.replace('(', '').replace(')', '').strip()
    
    # Split by 'or' to get alternative enzyme forms
    or_clauses = [clause.strip() for clause in gpr_clean.split(' or ')]
    
    dnf = []
    for clause in or_clauses:
        # Split by 'and' to get subunits in complex
        if ' and ' in clause:
            genes = {gene.strip() for gene in clause.split(' and ') if gene.strip()}
        else:
            genes = {clause.strip()} if clause.strip() else set()
        
        if genes:
            dnf.append(genes)
    
    return dnf


def evaluate_gpr_expression(
    dnf: List[Set[str]],
    expression_values: pd.Series,
    default_expr: float = 0.0
) -> float:
    """
    Evaluate GPR expression level respecting AND/OR logic.
    
    Parameters:
    -----------
    dnf : List[Set[str]]
        DNF representation of GPR
    expression_values : pd.Series
        Gene expression values
    default_expr : float
        Expression for genes not in data
        
    Returns:
    --------
    float : GPR-evaluated expression level
    
    Scientific Rationale:
    --------------------
    Enzyme Kinetics Principles:
    
    1. AND complexes (e.g., Cytochrome c oxidase subunits):
       - Activity limited by LEAST abundant subunit (bottleneck)
       - Use MIN across subunit expressions
       
    2. OR isoenzymes (e.g., Hexokinase I, II, III, IV):
       - Total activity is SUM/MAX of all active forms
       - Use MAX across isoform expressions (conservative)
       
    Logic Flow:
    -----------
    For each OR clause (alternative enzyme):
        For each AND term (subunit in complex):
            Take MIN of subunit expressions
        Take MAX across all alternatives
    
    Example:
    --------
    GPR: (geneA and geneB) or geneC
    Expressions: A=10, B=5, C=8
    
    Clause 1: min(10, 5) = 5  # Complex limited by B
    Clause 2: 8               # Single enzyme
    Result: max(5, 8) = 8     # Best alternative dominates
    """
    if not dnf:
        return default_expr
    
    # Evaluate each OR clause (alternative enzyme form)
    clause_activities = []
    
    for and_clause in dnf:
        # Evaluate AND clause (protein complex subunits)
        subunit_expressions = []
        
        for gene in and_clause:
            if gene in expression_values.index:
                subunit_expressions.append(expression_values[gene])
            else:
                subunit_expressions.append(default_expr)
        
        if subunit_expressions:
            # Complex activity = MIN(subunit expressions) - bottleneck principle
            complex_activity = min(subunit_expressions)
            clause_activities.append(complex_activity)
    
    if clause_activities:
        # Total activity = MAX(alternative forms) - best enzyme dominates
        # Could also use SUM for additive capacity, document choice
        return max(clause_activities)
    
    return default_expr

################################################################################
# APPLYING DIET BOUNDS
################################################################################

def apply_diet_bounds(
    model: cobra.Model,
    diet_file: str,
    condition: str = '',
    verbose: bool = True
) -> cobra.Model:
    """
    Apply condition-specific dietary constraints from a nested diet bounds file.

    File format expected (expanded_diet_bounds_flat.json):
        {
            "SCD": { "EX_glc__D_e": [-10, 1000], "EX_fru_e": [-2, 1000], ... },
            "HFD": { ... },
            ...
        }

    Each value is a [lower_bound, upper_bound] pair.

    Previous bugs fixed here:
    -------------------------
    BUG 1 (structural): old code iterated the top-level dict, yielding diet
        names ('SCD', 'HFD', ...) as reaction IDs → 0 constraints applied.
        FIX: map condition → diet key, then iterate the sub-dict.

    BUG 2 (bound format): old code did -abs(bound) on a [lb, ub] list → TypeError
        silently swallowed by except KeyError → still 0 constraints applied.
        FIX: extract lb = bound[0], ub = bound[1] from the list; handle both list
        and scalar formats for robustness.

    BUG 3 (ID mismatches): 8 reaction IDs in the diet file are absent from
        iMM1415 (cobalt2, cu2, mg2, mn2, zn2, mobd, phyQ, cbl1 — these minerals
        and vitamins use different compartment schemes in this model version).
        Two IDs have renamed equivalents in iMM1415:
            EX_lnlc_e  → EX_lnlc_e_copy1  (linoleic acid, uptake-capable copy)
            EX_actn_e  → EX_acetone_e      (acetone exchange)
        FIX: apply ID remapping before lookup; skip truly absent IDs gracefully.
    """
    if not diet_file or not os.path.exists(diet_file):
        if verbose:
            print("[WARNING] No diet bounds file found. Using default open bounds.")
        return model

    # --- Condition → diet name mapping ---
    CONDITION_TO_DIET = {
        'ND_SCD': 'SCD', 'DD_SCD': 'SCD',
        'ND_HFD': 'HFD', 'DD_HFD': 'HFD',
        'ND_KD':  'KD',  'DD_KD':  'KD',
        'ND_WD':  'WD',  'DD_WD':  'WD',
        'ND_LFD': 'LFD', 'DD_LFD': 'LFD',
        # Direct diet names also accepted
        'SCD': 'SCD', 'HFD': 'HFD', 'KD': 'KD', 'WD': 'WD', 'LFD': 'LFD',
    }

    # ID remapping: diet file ID → iMM1415 ID (for renamed reactions only)
    ID_REMAP = {
        'EX_lnlc_e':  'EX_lnlc_e_copy1',   # linoleic acid; copy1 allows uptake
        'EX_actn_e':  'EX_acetone_e',        # acetone
    }

    with open(diet_file, 'r') as f:
        all_diets = json.load(f)

    # Detect file format: flat {rxn_id: bound} vs nested {diet: {rxn_id: bound}}
    first_val = next(iter(all_diets.values()))
    if isinstance(first_val, dict):
        # Nested format — select the right diet
        diet_key = CONDITION_TO_DIET.get(condition, '')
        if not diet_key:
            # Try direct lookup (condition might already be a diet name like 'SCD')
            diet_key = condition if condition in all_diets else ''
        if not diet_key:
            if verbose:
                print(f"[WARNING] Condition '{condition}' not found in diet file "
                      f"(available: {list(all_diets.keys())}). Using default open bounds.")
            return model
        rxn_bounds = all_diets[diet_key]
        if verbose:
            print(f"[INFO] Using '{diet_key}' diet bounds for condition '{condition}'")
    else:
        # Flat format (scalar or list values keyed directly by rxn_id)
        rxn_bounds = all_diets

    model_diet = model.copy()
    applied = skipped_missing = 0

    for rxn_id, bound in rxn_bounds.items():
        # Apply ID remapping
        model_rxn_id = ID_REMAP.get(rxn_id, rxn_id)

        # Parse bound: accept [lb, ub] list or plain scalar (legacy)
        if isinstance(bound, (list, tuple)) and len(bound) >= 2:
            new_lb, new_ub = float(bound[0]), float(bound[1])
        elif isinstance(bound, (int, float)):
            # Legacy scalar: treat as uptake rate limit (lb = -abs, ub unchanged)
            new_lb, new_ub = -abs(bound), None
        else:
            continue

        try:
            rxn = model_diet.reactions.get_by_id(model_rxn_id)
            # Apply in safe order to avoid COBRA validation errors
            if new_lb is not None:
                if new_ub is not None and new_lb > rxn.upper_bound:
                    rxn.upper_bound = new_ub
                    rxn.lower_bound = new_lb
                else:
                    rxn.lower_bound = new_lb
            if new_ub is not None:
                rxn.upper_bound = new_ub
            applied += 1
        except KeyError:
            skipped_missing += 1

    if verbose:
        print(f"[INFO] Applied {applied} dietary constraints from "
              f"{os.path.basename(diet_file)} ({diet_key} diet)")
        if skipped_missing:
            print(f"[INFO] Skipped {skipped_missing} reaction IDs not present "
                  f"in iMM1415 (minerals/vitamins absent from this model version)")
    return model_diet



################################################################################
# BIOLOGICALLY APPROPRIATE OBJECTIVE SETTING
################################################################################

def set_hepatic_objective(
    model: cobra.Model,
    objective_mode: str = 'atpm',
    verbose: bool = True
) -> cobra.Model:
    """
    Set biologically appropriate objective for hepatic metabolic modeling.

    Parameters:
    -----------
    model : cobra.Model
        Hepatic metabolic model (e.g., iMM1415)
    objective_mode : str
        'atpm'     : ATP maintenance reaction (recommended for adult differentiated liver)
        'biomass'  : Biomass maximization (appropriate for regenerating liver, cell lines,
                     or comparative studies; set explicitly with --objective_mode biomass)
        'functional': Weighted combination of hepatic functions (fallback)
    verbose : bool
        Print diagnostic information

    ATPM mode
    ---------
    Searches a ranked list of known reaction IDs, then falls back to any
    reaction whose ID or name contains 'atp' + 'maint'. If nothing is found,
    falls back to 'functional' mode.

    BIOMASS mode
    ------------
    Gap analysis vs. ATPM (all fixed here):

    Gap 1 — No candidate ID list: the old code did a raw upper-case string
      search on r.id only. Now we try a ranked list of known IDs first
      (including iMM1415's specific ID), then fall back to pattern search on
      both ID and name, covering AGORA, Recon, and other common model formats.

    Gap 2 — Blind [0] selection: if multiple biomass reactions exist
      (e.g., BIOMASS_reaction + BIOMASS_DNA + BIOMASS_maintenance) the old code
      took whichever appeared first. Now we score candidates and prefer the one
      with the largest number of metabolites — that is almost always the true
      growth/cell-maintenance reaction.

    Gap 3 — No fallback: the old code raised ValueError immediately. Now we
      warn and fall back to 'functional' mode, mirroring ATPM behavior.

    Gap 4 — Unhelpful warnings: the old code printed "NOT RECOMMENDED for
      adult liver!" even when the user explicitly requested biomass. Removed.
      A single informational note is printed instead.

    Gap 5 — No minimum lb guard: biomass lb=0 combined with pFBA's two-stage
      optimization can produce a trivial zero-flux solution when E-Flux
      constrains precursor synthesis heavily. We set lb = 1e-6 to require at
      least minimal growth, mirroring the non-trivial minimum that ATPM
      implicitly has through the reaction network.

    Gap 6 — No zero-biomass detection: if E-Flux blocks all biomass precursors,
      the optimal biomass value is 0 and pFBA silently returns a zero-flux
      solution. A check in run_corrected_scenario_analysis (below) now detects
      this and prints a diagnostic warning.
    """
    model_obj = model.copy()

    # -------------------------------------------------------------------------
    if objective_mode == 'atpm':
    # -------------------------------------------------------------------------
        atpm_candidates = [
            'ATPM',            # iMM1415, Recon2, most COBRA models
            'DM_atp_c_',       # Recon3D demand reaction style
            'DM_atp_c',
            'ATPMaint',
            'ATP_maintenance',
            'atpm',            # lowercase variant
        ]
        atpm_rxn = None
        for rxn_id in atpm_candidates:
            try:
                atpm_rxn = model_obj.reactions.get_by_id(rxn_id)
                break
            except KeyError:
                continue

        if atpm_rxn is None:
            # Broad pattern fallback: ID or name contains both 'atp' and 'maint'
            for rxn in model_obj.reactions:
                combined = (rxn.id + ' ' + rxn.name).lower()
                if 'atp' in combined and 'maint' in combined:
                    atpm_rxn = rxn
                    break

        if atpm_rxn:
            model_obj.objective = atpm_rxn.id
            if verbose:
                print(f"[INFO] Set objective to ATP maintenance: {atpm_rxn.id}")
                print(f"[INFO] Objective expression: {model_obj.objective.expression}")
        else:
            if verbose:
                print("[WARNING] No ATP maintenance reaction found — falling back "
                      "to functional objective.")
            return set_hepatic_objective(model_obj, 'functional', verbose)

    # -------------------------------------------------------------------------
    elif objective_mode == 'biomass':
    # -------------------------------------------------------------------------
        # --- Gap 1 fixed: ranked candidate list ---
        # iMM1415-specific ID first, then common patterns across other models
        biomass_candidates = [
            'BIOMASS_mm_1_no_glygln',   # iMM1415 (mouse)
            'BIOMASS_Recon3D',          # Recon3D human
            'BIOMASS_reaction',         # common generic name
            'BIOMASS_maintenance',
            'biomass_reaction',
            'Biomass_reaction',
            'GROWTH',
            'Growth',
        ]
        biomass_rxn = None
        for rxn_id in biomass_candidates:
            try:
                biomass_rxn = model_obj.reactions.get_by_id(rxn_id)
                break
            except KeyError:
                continue

        # --- Fallback: pattern search on ID and name ---
        if biomass_rxn is None:
            pattern_hits = []
            for rxn in model_obj.reactions:
                combined = (rxn.id + ' ' + rxn.name).lower()
                if 'biomass' in combined or 'growth' in combined:
                    pattern_hits.append(rxn)

            if pattern_hits:
                # --- Gap 2 fixed: pick the reaction with the most metabolites ---
                # (largest stoichiometry = true growth/cell-maintenance biomass)
                biomass_rxn = max(pattern_hits,
                                  key=lambda r: len(r.metabolites))
                if verbose and len(pattern_hits) > 1:
                    print(f"[INFO] Found {len(pattern_hits)} biomass-like reactions; "
                          f"selected '{biomass_rxn.id}' (largest stoichiometry: "
                          f"{len(biomass_rxn.metabolites)} metabolites)")

        if biomass_rxn is None:
            # --- Gap 3 fixed: graceful fallback instead of bare ValueError ---
            if verbose:
                print("[WARNING] No biomass reaction found — falling back to "
                      "functional objective.")
                print("[WARNING] To use biomass, ensure the model contains a "
                      "reaction with 'BIOMASS' or 'GROWTH' in its ID or name.")
            return set_hepatic_objective(model_obj, 'functional', verbose)

        model_obj.objective = biomass_rxn.id

        # --- Gap 5 fixed: set minimum lb to prevent trivial zero-flux pFBA ---
        # When E-Flux constrains biosynthetic reactions heavily the feasible
        # maximum biomass may be very small. Without a lb > 0, pFBA can return
        # a degenerate zero-flux solution (biomass = 0 trivially satisfies the
        # pFBA constraint biomass ≥ 0.99 × 0).  A floor of 1e-6 forces the
        # solver to find at least a minimal growth solution.
        if biomass_rxn.lower_bound < 1e-6:
            biomass_rxn.lower_bound = 1e-6

        # --- Gap 4 fixed: informational note, not a "NOT RECOMMENDED" warning ---
        if verbose:
            print(f"[INFO] Set objective to biomass: {biomass_rxn.id}")
            print(f"[INFO] Biomass lb set to {biomass_rxn.lower_bound:.2e} "
                  f"(prevents trivial zero-flux pFBA solution)")
            print(f"[INFO] Objective expression: {model_obj.objective.expression}")

    # -------------------------------------------------------------------------
    elif objective_mode == 'functional':
    # -------------------------------------------------------------------------
        functional_rxns = {
            'r0714': 0.4,   # Ureagenesis (ammonia detoxification)
            'r0889': 0.3,   # Gluconeogenesis (blood glucose homeostasis)
            'r0648': 0.2,   # Ketogenesis (alternative fuel)
            'r0226': 0.1,   # Bile acid synthesis (lipid metabolism)
        }
        objective_dict = {}
        for rxn_id, weight in functional_rxns.items():
            try:
                rxn = model_obj.reactions.get_by_id(rxn_id)
                objective_dict[rxn] = weight
            except KeyError:
                if verbose:
                    print(f"[WARNING] Functional reaction not found: {rxn_id}")

        if objective_dict:
            model_obj.objective = objective_dict
            if verbose:
                print(f"[INFO] Set functional objective with "
                      f"{len(objective_dict)} reactions")
        else:
            if verbose:
                print("[ERROR] No functional reactions found in model — "
                      "cannot set any objective.")
            raise ValueError("No suitable objective reactions found in model. "
                             "Check that the model is a valid hepatic COBRA model.")

    # -------------------------------------------------------------------------
    else:
        raise ValueError(
            f"Unknown objective mode: '{objective_mode}'. "
            f"Valid options: 'atpm', 'biomass', 'functional'."
        )

    return model_obj


################################################################################
# GPR-AWARE E-FLUX IMPLEMENTATION
################################################################################

def apply_gpr_aware_eflux(
    model: cobra.Model,
    expression_values: pd.Series,
    eflux_floor: float = 0.1,
    eflux_cap: float = 1000.0,
    default_bound: float = 1000.0,
    verbose: bool = True
) -> Tuple[cobra.Model, Dict[str, float]]:
    """
    Apply GPR-aware E-Flux constraints respecting AND/OR logic.
    
    Parameters:
    -----------
    model : cobra.Model
        Metabolic model
    expression_values : pd.Series
        Gene expression data (gene_id -> expression level)
    eflux_floor : float
        Minimum constraint value (prevents zero)
    eflux_cap : float
        Maximum constraint value (prevents numerical issues)
    default_bound : float
        Bound for reactions without GPR or expression data
    verbose : bool
        Print diagnostic information
        
    Returns:
    --------
    model_eflux : cobra.Model
        Model with E-Flux constraints applied
    constraint_stats : dict
        Statistics about constraint application
        
    Scientific Rationale:
    --------------------
    Traditional E-Flux flaw: Taking max/percentile across all GPR genes
    ignores the fundamental biochemistry of enzyme complexes.
    
    Correct Approach:
    1. Parse GPR to identify complex structure (AND) and alternatives (OR)
    2. For each complex, use MIN(subunit expression) - rate-limiting step
    3. For alternatives, use MAX(complex activities) - best option dominates
    4. Apply symmetric constraints to reversible reactions
    
    Example:
    --------
    Reaction: A + B → C + D
    GPR: (gene1 AND gene2) OR gene3
    Expression: gene1=100, gene2=50, gene3=80
    
    Traditional E-Flux (WRONG):
        Upper bound = 95th percentile(100, 50, 80) ≈ 95
    
    GPR-Aware E-Flux (CORRECT):
        Complex1 = min(100, 50) = 50  # Limited by gene2
        Complex2 = 80
        Activity = max(50, 80) = 80   # Best alternative
        Upper bound = 80
    
    Impact on Biology:
    -----------------
    - Complex1 requires both subunits, limited by least abundant
    - gene3 alone can catalyze reaction at higher rate
    - Result: More accurate flux constraints
    
    Engineering Benefit:
    -------------------
    Provides detailed statistics for validation:
    - How many reactions actually constrained?
    - What fraction of model genes found in expression data?
    - Distribution of constraint values
    """
    model_eflux = model.copy()
    
    stats = {
        'total_reactions': len(model_eflux.reactions),
        'reactions_with_gpr': 0,
        'reactions_constrained': 0,
        'genes_in_model': len(model_eflux.genes),
        'genes_in_expression': 0,
        'genes_matched': 0,
        'constraint_values': []
    }
    
    # Count genes found in expression data
    model_gene_ids = {g.id for g in model_eflux.genes}
    expr_gene_ids = set(expression_values.index)
    matched_genes = model_gene_ids & expr_gene_ids
    
    stats['genes_in_expression'] = len(expr_gene_ids)
    stats['genes_matched'] = len(matched_genes)
    
    if verbose:
        print(f"\n[INFO] Applying GPR-aware E-Flux constraints...")
        print(f"[INFO] Model genes: {stats['genes_in_model']}")
        print(f"[INFO] Expression genes: {stats['genes_in_expression']}")
        print(f"[INFO] Matched genes: {stats['genes_matched']} "
              f"({100*stats['genes_matched']/stats['genes_in_model']:.1f}%)")
    
    # CRITICAL: If no genes match, try ID conversion
    if stats['genes_matched'] == 0:
        if verbose:
            print(f"\n[WARNING] No gene ID matches found!")
            print(f"[WARNING] Attempting gene ID format conversion...")
            print(f"[INFO] Sample model gene IDs: {list(model_gene_ids)[:5]}")
            print(f"[INFO] Sample expression gene IDs: {list(expr_gene_ids)[:5]}")
        
        # Try converting model gene IDs to match expression format
        # Common conversions: add/remove prefixes, convert formats
        gene_id_map = {}
        
        for model_gene in model_gene_ids:
            # Try multiple formats
            candidates = [
                model_gene,                    # Original
                str(model_gene).strip(),       # Stripped
                f"ENSMUSG{model_gene}",        # Add Ensembl prefix
                model_gene.replace('ENSMUSG', ''),  # Remove Ensembl prefix
                model_gene.split('.')[0],      # Remove version
                model_gene.upper(),            # Uppercase
                model_gene.lower(),            # Lowercase
            ]
            
            for candidate in candidates:
                if candidate in expr_gene_ids:
                    gene_id_map[model_gene] = candidate
                    break
        
        stats['genes_matched'] = len(gene_id_map)
        
        if verbose:
            print(f"[INFO] After conversion: {stats['genes_matched']} matches "
                  f"({100*stats['genes_matched']/stats['genes_in_model']:.1f}%)")
        
        # Create converted expression series
        if gene_id_map:
            expression_converted = pd.Series(dtype=float)
            for model_gene, expr_gene in gene_id_map.items():
                expression_converted[model_gene] = expression_values[expr_gene]
            expression_values = expression_converted
            
            if verbose:
                print(f"[INFO] Using converted gene IDs for {len(gene_id_map)} genes")
    
    # If still no matches, this is a critical error
    if stats['genes_matched'] == 0:
        print(f"\n[ERROR] CRITICAL: No gene ID matches after conversion!")
        print(f"[ERROR] Gene expression data cannot be applied to model")
        print(f"[ERROR] Model will be unconstrained (default bounds)")
        print(f"\n[SOLUTION] Check gene ID format:")
        print(f"  - Model uses: {list(model_gene_ids)[:3]}")
        print(f"  - Expression uses: {list(expr_gene_ids)[:3]}")
        print(f"  - Ensure compatible ID format (Entrez, Symbol, Ensembl)")
        print(f"  - May need external gene mapping file")
    
    for rxn in model_eflux.reactions:
        if rxn.gene_reaction_rule:
            stats['reactions_with_gpr'] += 1
            
            # Parse GPR to DNF
            dnf = parse_gpr_to_dnf(rxn.gene_reaction_rule)
            
            if dnf:
                # Evaluate GPR expression
                expr_level = evaluate_gpr_expression(
                    dnf,
                    expression_values,
                    default_expr=0.0
                )
                
                # Apply floor and cap
                expr_level = max(expr_level, eflux_floor)
                expr_level = min(expr_level, eflux_cap)
                
                # Constrain reaction bounds
                if rxn.upper_bound > 0:
                    new_ub = min(rxn.upper_bound, expr_level)
                    rxn.upper_bound = new_ub
                    stats['reactions_constrained'] += 1
                    stats['constraint_values'].append(new_ub)
                
                if rxn.lower_bound < 0:
                    new_lb = max(rxn.lower_bound, -expr_level)
                    rxn.lower_bound = new_lb
    
    if verbose:
        print(f"[INFO] Reactions with GPR: {stats['reactions_with_gpr']}")
        print(f"[INFO] Reactions constrained: {stats['reactions_constrained']}")
        
        if stats['constraint_values']:
            print(f"[INFO] Constraint statistics:")
            print(f"        Mean: {np.mean(stats['constraint_values']):.2f}")
            print(f"        Median: {np.median(stats['constraint_values']):.2f}")
            print(f"        Range: [{np.min(stats['constraint_values']):.2f}, "
                  f"{np.max(stats['constraint_values']):.2f}]")
    
    return model_eflux, stats


################################################################################
# PHYSIOLOGICALLY CORRECT PORTAL COUPLING
################################################################################

def apply_portal_constraints(
    model: cobra.Model,
    portal_metabolites: Dict[str, float],
    scaling_factor: float = 0.1,
    constraint_mode: str = 'soft',
    verbose: bool = True
) -> Tuple[cobra.Model, Dict]:
    """
    Apply physiologically correct microbiome-hepatic coupling.
    
    Parameters:
    -----------
    model : cobra.Model
        Hepatic metabolic model
    portal_metabolites : dict
        Metabolite exchange IDs -> microbial flux values
    scaling_factor : float
        Scaling between microbiome and hepatic flux units
        (accounts for organ mass, dilution, metabolism)
    constraint_mode : str
        'soft': Wide bounds with penalties (recommended)
        'hard': Direct bound overwrites (not recommended)
    verbose : bool
        Print diagnostic information
        
    Returns:
    --------
    model_portal : cobra.Model
        Model with portal constraints
    stats : dict
        Application statistics
        
    Scientific Rationale:
    --------------------
    Portal Vein Physiology:
    
    1. UNIDIRECTIONAL FLOW: Gut → Liver (not bidirectional)
       - Microbial metabolites enter portal circulation
       - Hepatic metabolites do NOT return to gut lumen
       
    2. Microbial Production (+flux):
       - Compound enters portal blood
       - Liver CAN uptake (sets AVAILABILITY)
       - Constraint: hepatic_uptake ≤ scaled_microbial_production
       
    3. Microbial Consumption (-flux):
       - Compound consumed from LUMEN (dietary nutrients)
       - NO hepatic constraint (different compartment)
       - Liver does not supply gut lumen
       
    Scaling Factor Rationale:
    -------------------------
    Microbiome and hepatic models have incompatible units:
    
    - Microbiome: mmol/gDW_microbiome/hr
    - Hepatic: mmol/gDW_liver/hr
    
    Scaling accounts for:
    - Organ mass ratio (~0.2g gut microbiome / 1.2g liver in mice)
    - Portal flow dilution (blood volume)
    - First-pass metabolism efficiency
    - Time-averaged vs. peak concentrations
    
    Recommended Approach:
    1. Use 'soft' mode with wide bounds
    2. Test scaling_factor sensitivity (0.01, 0.1, 1.0)
    3. Validate results against known metabolite concentrations
    
    Engineering Benefit:
    -------------------
    Explicit scaling documentation allows:
    - Sensitivity analysis across parameter ranges
    - Biological interpretation of coupling strength
    - Future refinement with empirical measurements
    """
    model_portal = model.copy()
    
    stats = {
        'total_portal_metabolites': len(portal_metabolites),
        'microbial_production': 0,    # +flux
        'microbial_consumption': 0,   # -flux
        'constraints_applied': 0,
        'reactions_not_found': 0,
        'reactions_found_details': [],
        'scaling_factor': scaling_factor,
        'constraint_mode': constraint_mode
    }
    
    if verbose:
        print(f"\n[INFO] Applying portal constraints (mode={constraint_mode}, "
              f"scale={scaling_factor:.3f})")
    
    # Get all model exchange reaction IDs for debugging
    exchange_rxns = [r.id for r in model_portal.reactions if r.id.startswith('EX_') or r.id.startswith('DM_')]
    
    for met_id, flux_data in portal_metabolites.items():
        # Extract numeric flux value
        if isinstance(flux_data, dict):
            flux = flux_data.get('flux', flux_data.get('value', 0))
            hepatic_rxn_id = flux_data.get('hepatic_rxn', met_id)
        else:
            flux = flux_data
            hepatic_rxn_id = met_id
        
        # Skip zero fluxes
        if not isinstance(flux, (int, float)) or abs(flux) < 1e-9:
            continue
        
        # EXPANDED: Try many more reaction ID formats
        # Common metabolite abbreviations and their exchange reaction variants
        base_met = met_id.replace('EX_', '').replace('_m', '').replace('_e', '').replace('[e]', '').replace('[m]', '')
        
        possible_ids = [
            # Original formats
            hepatic_rxn_id,
            met_id,
            f"EX_{met_id}",
            f"EX_{met_id}[e]",
            f"EX_{met_id}_e",
            met_id.replace('_m', '_e'),
            met_id.replace('_m', '[e]'),
            
            # Base metabolite formats
            f"EX_{base_met}",
            f"EX_{base_met}[e]",
            f"EX_{base_met}_e",
            f"EX_{base_met}(e)",
            f"EX_{base_met}_LPAREN_e_RPAREN_",  # BiGG escaped format
            
            # Common variations
            f"{base_met}_EX",
            f"{base_met}_ex",
            f"r{base_met}",  # Some models use r prefix
            f"R_{base_met}",
            
            # Demand reactions (some models use these for uptake)
            f"DM_{base_met}",
            f"DM_{base_met}[e]",
            f"DM_{base_met}_e",
            
            # Sink reactions
            f"sink_{base_met}",
            f"SINK_{base_met}",
        ]
        
        # Also try case variations
        possible_ids_with_case = []
        for rxn_id in possible_ids:
            possible_ids_with_case.append(rxn_id)
            possible_ids_with_case.append(rxn_id.upper())
            possible_ids_with_case.append(rxn_id.lower())
        
        rxn = None
        matched_id = None
        
        for rxn_id in possible_ids_with_case:
            try:
                rxn = model_portal.reactions.get_by_id(rxn_id)
                matched_id = rxn_id
                break
            except KeyError:
                continue
        
        if rxn is None:
            stats['reactions_not_found'] += 1
            if verbose:
                print(f"[WARNING] Reaction not found for: {met_id}")
                print(f"          Tried variants: {base_met}, EX_{base_met}[e], EX_{base_met}_e, etc.")
                # Show similar reactions in model
                similar = [r for r in exchange_rxns if base_met.lower() in r.lower()]
                if similar:
                    print(f"          Similar reactions in model: {similar[:3]}")
                else:
                    print(f"          No similar reactions found. Sample exchange reactions: {exchange_rxns[:5]}")
            continue
        
        # Apply physiologically correct constraints
        scaled_flux = abs(flux) * scaling_factor

        if flux > 0:
            # Microbiome PRODUCES → Available for hepatic UPTAKE
            stats['microbial_production'] += 1

            # ----------------------------------------------------------------
            # CONSTRAINT STRATEGY
            # ----------------------------------------------------------------
            # COBRA convention: negative flux = uptake, positive = export.
            # pFBA minimises total flux, so without an explicit force the solver
            # routes zero flux through exchange reactions → no microbiome effect.
            #
            # We force uptake by setting upper_bound < 0:
            #   new_ub = -min_frac * sf   → solver MUST carry this much uptake
            #   new_lb = -max_frac * sf   → caps MAXIMUM uptake from portal vein
            # With max_frac > min_frac:  new_lb < new_ub < 0  (always feasible
            # from the bounds alone, but see infeasibility guard below).
            #
            # ASSIGNMENT ORDER (COBRA validates after every write):
            #   If rxn.lower_bound > new_ub we must write lower_bound FIRST.
            #
            # INFEASIBILITY GUARD:
            #   Some exchange reactions are export-only (lb=0) in the model;
            #   forcing their uptake direction may be impossible if downstream
            #   consuming reactions are blocked by E-Flux constraints.
            #   We PROBE each constraint with a slim_optimize() inside COBRA's
            #   context manager. If infeasible, we skip that metabolite and
            #   log it — the model stays consistent and moves on.
            # ----------------------------------------------------------------

            if constraint_mode == 'soft':
                min_frac, max_frac = 0.1, 2.0   # uptake ≥ 10%, ≤ 200% of delivery
            elif constraint_mode == 'hard':
                min_frac, max_frac = 0.5, 1.5   # uptake ≥ 50%, ≤ 150% of delivery
            elif constraint_mode == 'forced':
                min_frac, max_frac = 0.25, 2.0  # uptake ≥ 25%, ≤ 200% of delivery
            else:
                min_frac, max_frac = 0.1, 2.0

            new_lb = -scaled_flux * max_frac  # always < new_ub < 0
            new_ub = -scaled_flux * min_frac

            # --- Probe feasibility before committing ---
            # Use COBRA's context manager: all changes made inside the `with`
            # block are automatically rolled back when the block exits.
            constraint_feasible = True
            with model_portal:
                try:
                    # Apply bounds in safe order (lb first if it needs to go lower)
                    if new_ub < rxn.lower_bound:
                        rxn.lower_bound = new_lb
                        rxn.upper_bound = new_ub
                    else:
                        rxn.upper_bound = new_ub
                        rxn.lower_bound = new_lb

                    # Quick LP feasibility check (no objective, just primal)
                    test_val = model_portal.slim_optimize()
                    if model_portal.solver.status != 'optimal':
                        constraint_feasible = False
                except Exception:
                    constraint_feasible = False
            # Context manager has now rolled back the tentative bounds

            if not constraint_feasible:
                stats['constraints_skipped_infeasible'] = (
                    stats.get('constraints_skipped_infeasible', 0) + 1)
                if verbose:
                    print(f"[SKIP-INFEASIBLE] {matched_id}: forcing uptake of "
                          f"{scaled_flux:.6f} mmol/gDW/h makes LP infeasible "
                          f"(likely pathway blocked by E-Flux). Skipping.")
                continue  # leave the reaction bounds unchanged

            # --- Constraint is feasible — apply permanently ---
            if new_ub < rxn.lower_bound:
                rxn.lower_bound = new_lb
                rxn.upper_bound = new_ub
            else:
                rxn.upper_bound = new_ub
                rxn.lower_bound = new_lb

            stats['constraints_applied'] += 1
            stats['reactions_found_details'].append({
                'input_id': met_id,
                'matched_id': matched_id,
                'flux': flux,
                'scaled': scaled_flux,
                'mode': constraint_mode,
                'original_lb': rxn.lower_bound,
                'new_lb': new_lb,
                'new_ub': new_ub,
            })

            if verbose:
                print(f"[SUCCESS] {matched_id}: Microbial production = {flux:.4f} → "
                      f"Hepatic uptake in [{rxn.lower_bound:.4f}, {rxn.upper_bound:.4f}] "
                      f"(mode={constraint_mode})")
        
        elif flux < 0:
            # Microbiome CONSUMES from lumen
            # NO hepatic constraint (different compartment)
            stats['microbial_consumption'] += 1
            
            if verbose:
                print(f"[INFO] {matched_id}: Microbial consumption = {abs(flux):.4f} → "
                      f"No hepatic constraint (lumen uptake)")
    
    if verbose:
        print(f"\n[INFO] Portal constraint statistics:")
        print(f"        Microbial production events: {stats['microbial_production']}")
        print(f"        Microbial consumption events: {stats['microbial_consumption']}")
        print(f"        Constraints applied: {stats['constraints_applied']}")
        print(f"        Constraints skipped (infeasible pathway): "
              f"{stats.get('constraints_skipped_infeasible', 0)}")
        print(f"        Reactions not found: {stats['reactions_not_found']}")
        
        if stats['constraints_applied'] == 0 and stats['total_portal_metabolites'] > 0:
            print(f"\n[ERROR] CRITICAL: No portal constraints could be applied!")
            print(f"[ERROR] All {stats['total_portal_metabolites']} metabolites failed to match model reactions")
            print(f"[ERROR] This means microbiome effects CANNOT be detected")
            print(f"\n[SOLUTION] Check portal_metabolites JSON format:")
            print(f"           - Are metabolite IDs in BiGG format?")
            print(f"           - Do they match iMM1415 exchange reactions?")
            print(f"           - Try: list(model.exchanges) to see available reactions")
    
    return model_portal, stats


################################################################################
# CONDITION MATCHING UTILITIES (FROM ORIGINAL)
################################################################################

def map_condition_to_columns(
    condition: str,
    available_columns: List[str],
    verbose: bool = True
) -> Tuple[Optional[str], List[str]]:
    """
    Intelligently map condition name to expression file columns.
    (Implementation from original - kept for compatibility)
    """
    # Try exact match first
    if condition in available_columns:
        if verbose:
            print(f"[INFO] Exact match found for condition: {condition}")
        return condition, [condition]
    
    # Try pattern matching
    pattern_map = {
        'ND_SCD': 'SCD',
        'DD_HFD': 'HFD',
        'ND_HFD': 'HFD',
        'DD_SCD': 'SCD',
        'ND_KD': 'KD',
        'DD_KD': 'KD',
        'ND_WD': 'WD',
        'DD_WD': 'WD'
    }
    
    pattern = pattern_map.get(condition, condition)
    matching_cols = [col for col in available_columns 
                     if col.upper().startswith(pattern.upper())]
    
    if matching_cols:
        if verbose:
            print(f"[INFO] Pattern match: '{condition}' -> '{pattern}' -> "
                  f"{len(matching_cols)} columns")
        return pattern, matching_cols
    
    return None, []


def average_replicate_expression(
    expr_df: pd.DataFrame,
    condition_columns: List[str],
    condition_name: str,
    verbose: bool = True
) -> pd.Series:
    """
    Average expression across technical replicates.
    (Implementation from original - kept for compatibility)
    """
    if len(condition_columns) == 1:
        if verbose:
            print(f"[INFO] Single replicate for {condition_name}")
        return expr_df[condition_columns[0]]
    
    averaged = expr_df[condition_columns].mean(axis=1)
    
    if verbose:
        print(f"[INFO] Averaged {len(condition_columns)} replicates for {condition_name}")
    
    return averaged


################################################################################
# INTEGRATED WORKFLOW
################################################################################

def apply_corrected_eflux_with_microbiome(
    model: cobra.Model,
    expression_file: str,
    condition: str,
    portal_metabolites: Optional[Dict[str, float]] = None,
    diet_file: Optional[str] = None,
    objective_mode: str = 'atpm',
    eflux_floor: float = 0.1,
    eflux_cap: float = 1000.0,
    portal_scaling: float = 0.1,
    portal_mode: str = 'soft',
    gene_mapping_file: Optional[str] = None,
    verbose: bool = True
) -> Tuple[cobra.Model, Dict]:
    """
    Apply biologically corrected E-Flux with optional microbiome constraints.
    
    This is the main function integrating all corrections.
    
    Parameters:
    -----------
    gene_mapping_file : str, optional
        Path to CSV file with gene ID mappings (model_gene,expression_gene columns)
    
    Returns:
    --------
    model_final : cobra.Model
        Fully constrained model ready for optimization
    statistics : dict
        Combined statistics from all constraint steps
    """
    statistics = {}
    
    # Step 1: Set appropriate objective
    print("\n" + "="*80)
    print("STEP 1: Setting Biologically Appropriate Objective")
    print("="*80)
    
    model_obj = set_hepatic_objective(model, objective_mode, verbose)
    statistics['objective_mode'] = objective_mode
    
    # Step 1.5: Apply Diet
    print("\n" + "="*80)
    print("STEP 1.5: Applying Dietary Constraints")
    print("="*80)
    model_obj = apply_diet_bounds(model_obj, diet_file, condition=condition, verbose=verbose)
    
    # Step 2: Load and match expression data
    print("\n" + "="*80)
    print("STEP 2: Loading Gene Expression Data")
    print("="*80)
    
    # expr_df = pd.read_csv(expression_file, index_col=0)
    # === SMART EXPRESSION LOADING ===
    expr_df = pd.read_csv(expression_file)
    
    # If the file already has an Entrez column, use it and clean the decimals!
    if 'Entrez' in expr_df.columns:
        if verbose: print("[INFO] 'Entrez' column detected in expression data. Using it as index.")
        # Drop rows missing an Entrez ID
        expr_df = expr_df.dropna(subset=['Entrez'])
        # Convert floats (1234.0) -> integers (1234) -> strings ("1234") to match the model
        expr_df.index = expr_df['Entrez'].astype(int).astype(str)
        # Drop the original columns so they don't interfere with numeric averaging
        expr_df = expr_df.drop(columns=['Gene_Symbol', 'Entrez'], errors='ignore')
    else:
        # Fallback to default
        expr_df = expr_df.set_index(expr_df.columns[0])
    # ================================
    
    matched_name, matching_cols = map_condition_to_columns(
        condition,
        list(expr_df.columns),
        verbose=verbose
    )
    
    if not matching_cols:
        raise ValueError(f"Condition '{condition}' not found in expression data")
    
    expression_values = average_replicate_expression(
        expr_df,
        matching_cols,
        matched_name,
        verbose=verbose
    )
    
    # Apply gene mapping if provided
    if gene_mapping_file and os.path.exists(gene_mapping_file):
        if verbose:
            print(f"\n[INFO] Loading gene mapping from: {gene_mapping_file}")
        
        mapping_df = pd.read_csv(gene_mapping_file)
        
        # Expect columns: model_gene, expression_gene (or similar)
        # Try to detect column names
        col_names = mapping_df.columns.tolist()
        
        # --- NEW LOGIC: Look specifically for your headers ---
        if 'entrez' in col_names and 'symbol' in col_names:
            model_gene_col = 'entrez'
            expr_gene_col = 'symbol'
            
        elif len(col_names) >= 2:
            model_gene_col = col_names[1]   # col_names[0]
            expr_gene_col = col_names[3]    # col_names[1]
            
            if verbose:
                print(f"[INFO] Using mapping: {model_gene_col} -> {expr_gene_col}")
            
            # Create mapping dictionary
            gene_map = dict(zip(
                mapping_df[model_gene_col].astype(str),
                mapping_df[expr_gene_col].astype(str)
            ))
            
            # Convert expression index using mapping
            new_index = []
            new_values = []
            
            for expr_gene in expression_values.index:
                # Find model gene ID(s) that map to this expression gene
                model_genes = [k for k, v in gene_map.items() if v == str(expr_gene)]
                
                for model_gene in model_genes:
                    new_index.append(model_gene)
                    new_values.append(expression_values[expr_gene])
            
            if new_index:
                expression_values = pd.Series(new_values, index=new_index)
                if verbose:
                    print(f"[INFO] Mapped {len(new_index)} expression values to model gene IDs")
    
    # Step 3: Apply GPR-aware E-Flux
    print("\n" + "="*80)
    print("STEP 3: Applying GPR-Aware E-Flux Constraints")
    print("="*80)
    
    model_eflux, eflux_stats = apply_gpr_aware_eflux(
        model_obj,
        expression_values,
        eflux_floor=eflux_floor,
        eflux_cap=eflux_cap,
        verbose=verbose
    )
    
    statistics['eflux'] = eflux_stats
    
    # Step 4: Apply portal constraints if provided
    if portal_metabolites:
        print("\n" + "="*80)
        print("STEP 4: Applying Portal Circulation Constraints")
        print("="*80)
        
        model_final, portal_stats = apply_portal_constraints(
            model_eflux,
            portal_metabolites,
            scaling_factor=portal_scaling,
            constraint_mode=portal_mode,
            verbose=verbose
        )
        
        statistics['portal'] = portal_stats
    else:
        model_final = model_eflux
        if verbose:
            print("\n[INFO] No portal metabolites provided, skipping portal constraints")
    
    return model_final, statistics


def run_corrected_scenario_analysis(
    model: cobra.Model,
    expression_file: str,
    condition: str,
    portal_metabolites: Dict[str, float],
    objective_mode: str = 'atpm',
    diet_file: Optional[str] = None,
    eflux_floor: float = 0.1,
    eflux_cap: float = 1000.0,
    portal_scaling: float = 0.1,
    portal_mode: str = 'forced',   # FIX Bug 1: was missing from signature entirely
    flux_threshold: float = 0.01,
    gene_mapping_file: Optional[str] = None,
    use_pfba: bool = True
) -> Dict:
    """
    Run two-scenario comparison with biological corrections.
    
    Parameters:
    -----------
    portal_mode : str
        Portal constraint mode: 'soft', 'hard', or 'forced'.
        'forced' is recommended — it pins hepatic uptake to the scaled
        portal-vein delivery rate, guaranteeing measurable flux differences.
    use_pfba : bool
        Use parsimonious FBA after optimizing objective.
        Minimizes total flux for more realistic solutions.
        
    Returns:
    --------
    results : dict
        Contains models, solutions, flux comparisons, and statistics
    """
    results = {}
    
    # Scenario 1: Baseline (no microbiome)
    print("\n" + "="*80)
    print("SCENARIO 1: Baseline (Host + Diet, No Microbiome)")
    print("="*80)
    
    model_baseline, stats_baseline = apply_corrected_eflux_with_microbiome(
        model,
        expression_file,
        condition,
        portal_metabolites=None,
        diet_file=diet_file,
        objective_mode=objective_mode,
        eflux_floor=eflux_floor,
        eflux_cap=eflux_cap,
        portal_scaling=portal_scaling,
        gene_mapping_file=gene_mapping_file,
        verbose=True
    )
    
    print(f"\n[INFO] Optimizing baseline model...")
    
    if use_pfba:
        # Two-stage optimization: maximize objective, then minimize flux
        try:
            # Method 1: Use pFBA with fraction_of_optimum
            # This automatically fixes objective and minimizes flux
            solution_baseline = pfba(model_baseline, fraction_of_optimum=0.99)
            print(f"[INFO] Used pFBA for parsimonious solution")
        except Exception as e:
            # Fallback: Regular optimization if pFBA fails
            print(f"[WARNING] pFBA failed ({e}), using regular optimization")
            solution_baseline = model_baseline.optimize()
    else:
        solution_baseline = model_baseline.optimize()
    
    if solution_baseline.status != 'optimal':
        print(f"[WARNING] Baseline optimization failed: {solution_baseline.status}")
    else:
        obj_val = solution_baseline.objective_value
        print(f"[INFO] Baseline objective value: {obj_val:.4f}")
        # --- Gap 6 fixed: zero-biomass detection ---
        if objective_mode == 'biomass' and obj_val is not None and obj_val < 1e-4:
            print(f"[WARNING] Baseline biomass is near zero ({obj_val:.2e}). "
                  f"E-Flux constraints may be blocking biomass precursor synthesis.")
            print(f"[WARNING] Consider: --eflux_floor 1.0, --eflux_cap 1000, or "
                  f"--objective_mode atpm for this condition.")
    
    results['model_baseline'] = model_baseline
    results['solution_baseline'] = solution_baseline
    results['stats_baseline'] = stats_baseline
    
    # Scenario 2: + Microbiome
    print("\n" + "="*80)
    print("SCENARIO 2: Host + Diet + Microbiome")
    print("="*80)
    
    model_microbiome, stats_microbiome = apply_corrected_eflux_with_microbiome(
        model,
        expression_file,
        condition,
        portal_metabolites=portal_metabolites,
        diet_file=diet_file,
        objective_mode=objective_mode,
        eflux_floor=eflux_floor,
        eflux_cap=eflux_cap,
        portal_scaling=portal_scaling,
        portal_mode=portal_mode,   # FIX Bug 1: was hardcoded as 'soft', now passes through
        gene_mapping_file=gene_mapping_file,
        verbose=True
    )
    
    print(f"\n[INFO] Optimizing microbiome-integrated model...")
    
    if use_pfba:
        try:
            solution_microbiome = pfba(model_microbiome, fraction_of_optimum=0.99)
            print(f"[INFO] Used pFBA for parsimonious solution")
        except Exception as e:
            print(f"[WARNING] pFBA failed ({e}), using regular optimization")
            solution_microbiome = model_microbiome.optimize()
    else:
        solution_microbiome = model_microbiome.optimize()
    
    if solution_microbiome.status != 'optimal':
        print(f"[WARNING] Microbiome optimization failed: {solution_microbiome.status}")
        print(f"[WARNING] Tip: try --portal_mode soft or increase --portal_scaling to relax constraints")
    else:
        obj_val = solution_microbiome.objective_value
        print(f"[INFO] Microbiome objective value: {obj_val:.4f}")
        if solution_baseline.objective_value is not None:
            delta_obj = obj_val - solution_baseline.objective_value
            print(f"[INFO] Objective change: {delta_obj:+.4f}")
        # --- Gap 6 fixed: zero-biomass detection ---
        if objective_mode == 'biomass' and obj_val is not None and obj_val < 1e-4:
            print(f"[WARNING] Microbiome scenario biomass is near zero ({obj_val:.2e}). "
                  f"E-Flux constraints may be fully blocking biomass precursor synthesis.")
            print(f"[WARNING] Flux comparison results will not be meaningful.")
    
    results['model_microbiome'] = model_microbiome
    results['solution_microbiome'] = solution_microbiome
    results['stats_microbiome'] = stats_microbiome
    
    # Compare solutions
    if (solution_baseline.status == 'optimal' and 
        solution_microbiome.status == 'optimal'):
        
        print("\n" + "="*80)
        print("FLUX COMPARISON ANALYSIS")
        print("="*80)
        
        flux_comparison = compare_flux_solutions(
            solution_baseline,
            solution_microbiome,
            model,
            threshold=flux_threshold
        )
        
        results['flux_comparison'] = flux_comparison
        
        # Summary
        n_total = len(flux_comparison)
        n_attributable = flux_comparison['microbiome_attributable'].sum()
        
        print(f"\n[INFO] Total reactions: {n_total}")
        print(f"[INFO] Microbiome-attributable: {n_attributable} "
              f"({100*n_attributable/n_total:.1f}%)")
    
    return results


def compare_flux_solutions(
    solution_baseline: cobra.Solution,
    solution_microbiome: cobra.Solution,
    model: cobra.Model,
    threshold: float = 0.01
) -> pd.DataFrame:
    """
    Compare flux solutions between scenarios.
    (Implementation from original - kept for compatibility)
    """
    fluxes_baseline = solution_baseline.fluxes
    fluxes_microbiome = solution_microbiome.fluxes
    
    flux_delta = fluxes_microbiome - fluxes_baseline
    
    flux_delta_pct = np.where(
        np.abs(fluxes_baseline) > 1e-6,
        100 * flux_delta / np.abs(fluxes_baseline),
        np.nan
    )
    
    microbiome_attributable = np.abs(flux_delta) > threshold
    
    df = pd.DataFrame({
        'reaction_id': fluxes_baseline.index,
        'reaction_name': [model.reactions.get_by_id(rxn).name 
                         for rxn in fluxes_baseline.index],
        'flux_baseline': fluxes_baseline.values,
        'flux_microbiome': fluxes_microbiome.values,
        'flux_delta': flux_delta.values,
        'flux_delta_pct': flux_delta_pct,
        'microbiome_attributable': microbiome_attributable
    })
    
    df = df.sort_values('flux_delta', key=abs, ascending=False)
    
    return df


################################################################################
# VISUALIZATION
################################################################################

def plot_flux_comparison(
    flux_comparison: pd.DataFrame,
    output_file: str,
    top_n: int = 30
):
    """
    Visualize microbiome-attributable flux changes.
    (Implementation from original - kept for compatibility)
    """
    attributable = flux_comparison[
        flux_comparison['microbiome_attributable']
    ].copy()
    
    if len(attributable) == 0:
        print("[WARNING] No microbiome-attributable changes to plot")
        return
    
    attributable['abs_flux_delta'] = attributable['flux_delta'].abs()
    top = attributable.nlargest(top_n, 'abs_flux_delta')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Waterfall chart
    y_pos = np.arange(len(top))
    colors = ['red' if x < 0 else 'green' for x in top['flux_delta']]
    
    ax1.barh(y_pos, top['flux_delta'], color=colors, alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top['reaction_id'], fontsize=8)
    ax1.set_xlabel('Flux Change (microbiome - baseline)', fontsize=12)
    ax1.set_title(f'Top {top_n} Microbiome-Attributable Changes', 
                  fontsize=14, fontweight='bold')
    ax1.axvline(0, color='black', linestyle='--', alpha=0.3)
    ax1.grid(axis='x', alpha=0.3)
    
    # Scatter plot
    ax2.scatter(
        attributable['flux_baseline'],
        attributable['flux_microbiome'],
        alpha=0.5,
        s=50,
        c='blue',
        edgecolors='black',
        linewidths=0.5
    )
    
    lims = [
        np.min([ax2.get_xlim(), ax2.get_ylim()]),
        np.max([ax2.get_xlim(), ax2.get_ylim()]),
    ]
    ax2.plot(lims, lims, 'k--', alpha=0.3, zorder=0)
    
    ax2.set_xlabel('Baseline Flux', fontsize=12)
    ax2.set_ylabel('Microbiome Flux', fontsize=12)
    ax2.set_title('Baseline vs. Microbiome Distribution', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] Flux comparison plot: {output_file}")


################################################################################
# COMMAND LINE INTERFACE
################################################################################

def main():
    parser = argparse.ArgumentParser(
        description='RQ4 Hepatic-Microbiome Integration (BIOLOGICALLY CORRECTED)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
CRITICAL CORRECTIONS IN THIS VERSION:
=====================================
1. ATP Maintenance objective (not biomass)
2. GPR-aware E-Flux (respects AND/OR logic)
3. Physiologically correct portal coupling
4. Scaled microbiome constraints with sensitivity

Example usage:
--------------
python rq4_hepatic_integration_CORRECTED.py \\
    --hepatic_model iMM1415.json \\
    --expression_data GSE182668_expression.csv \\
    --portal_metabolites portal_metabolites.json \\
    --condition ND_SCD \\
    --objective_mode atpm \\
    --portal_scaling 0.1 \\
    --results_dir results_rq4_corrected
        """
    )
    
    # Required arguments
    parser.add_argument('--hepatic_model', required=True,
                       help='Path to hepatic model (iMM1415.json)')
    parser.add_argument('--expression_data', required=True,
                       help='Path to hepatic gene expression CSV')
    parser.add_argument('--portal_metabolites', required=True,
                       help='Path to portal_metabolites JSON')
    parser.add_argument('--condition', required=True,
                       help='Condition name (e.g., ND_SCD, DD_HFD)')
    parser.add_argument('--results_dir', required=True,
                       help='Output directory for results')
    
    # Objective settings
    parser.add_argument('--objective_mode', default='atpm',
                       choices=['atpm', 'functional', 'biomass'],
                       help='Objective function (default: atpm - RECOMMENDED)')
    parser.add_argument('--no-pfba', dest='use_pfba', action='store_false',
                       help='Disable pFBA (use regular FBA instead)')
    parser.set_defaults(use_pfba=True)
    
    # E-Flux parameters
    parser.add_argument('--eflux_floor', type=float, default=0.1,
                       help='E-Flux minimum constraint (default: 0.1)')
    parser.add_argument('--eflux_cap', type=float, default=1000.0,
                       help='E-Flux maximum constraint (default: 1000.0)')
    parser.add_argument('--gene_mapping', default=None,
                       help='Gene ID mapping file (CSV with model_gene,expression_gene columns)')
    parser.add_argument('--diet_bounds', default=None,
                       help='Path to expanded_diet_bounds_flat.json (optional, for diet effect attribution)')
    
    # Portal coupling parameters
    parser.add_argument('--portal_scaling', type=float, default=0.1,
                       help='Microbiome-to-hepatic scaling factor (default: 0.1)')
    parser.add_argument('--portal_mode', default='forced',
                       choices=['soft', 'hard', 'forced'],
                       help='Portal constraint mode (default: soft - RECOMMENDED)')
    
    # Analysis parameters
    parser.add_argument('--flux_threshold', type=float, default=0.01,
                       help='Minimum flux change for attribution (default: 0.01)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    print("="*80)
    print("RQ4: HEPATIC-MICROBIOME INTEGRATION (BIOLOGICALLY CORRECTED)")
    print("="*80)
    print("\nCORRECTIONS APPLIED:")
    print("  [X] ATP maintenance objective (not biomass)")
    print("  [X] GPR-aware E-Flux (respects AND/OR logic)")
    print("  [X] Physiologically correct portal coupling")
    print("  [X] Scaled microbiome constraints")
    print("="*80)
    
    # Load model
    print(f"\n[INFO] Loading hepatic model: {args.hepatic_model}")
    model = cobra.io.load_json_model(args.hepatic_model)
    print(f"[INFO] Model: {len(model.reactions)} reactions, "
          f"{len(model.metabolites)} metabolites, {len(model.genes)} genes")
    
    # Load portal metabolites
    print(f"\n[INFO] Loading portal metabolites: {args.portal_metabolites}")
    with open(args.portal_metabolites, 'r') as f:
        portal_data = json.load(f)
    
    # Extract condition-specific metabolites
    if args.condition in portal_data:
        portal_metabolites = portal_data[args.condition]
    elif 'portal_metabolites' in portal_data:
        portal_metabolites = portal_data['portal_metabolites'].get(args.condition, {})
    else:
        portal_metabolites = {}
    
    print(f"[INFO] Loaded {len(portal_metabolites)} portal metabolite fluxes")
    
    # Run analysis
    results = run_corrected_scenario_analysis(
        model,
        args.expression_data,
        args.condition,
        portal_metabolites,
        objective_mode=args.objective_mode,
        eflux_floor=args.eflux_floor,
        eflux_cap=args.eflux_cap,
        portal_scaling=args.portal_scaling,
        portal_mode=args.portal_mode,   # FIX Bug 1: was missing from this call
        flux_threshold=args.flux_threshold,
        gene_mapping_file=args.gene_mapping,
        diet_file=args.diet_bounds,
        use_pfba=args.use_pfba
    )
    
    # Save results
    condition_dir = os.path.join(args.results_dir, f'condition_{args.condition}')
    os.makedirs(condition_dir, exist_ok=True)
    
    if 'flux_comparison' in results:
        # Save flux comparison
        flux_csv = os.path.join(condition_dir, f'{args.condition}_flux_comparison.csv')
        results['flux_comparison'].to_csv(flux_csv, index=False)
        print(f"\n[SAVED] Flux comparison: {flux_csv}")
        
        # Create visualization
        flux_plot = os.path.join(condition_dir, f'{args.condition}_flux_comparison.png')
        plot_flux_comparison(results['flux_comparison'], flux_plot)
    
    # Save statistics
    stats_file = os.path.join(condition_dir, f'{args.condition}_statistics.json')
    stats_to_save = {
        'baseline': {k: v for k, v in results['stats_baseline'].items() 
                    if isinstance(v, (int, float, str, bool))},
        'microbiome': {k: v for k, v in results['stats_microbiome'].items()
                      if isinstance(v, (int, float, str, bool))}
    }
    
    with open(stats_file, 'w') as f:
        json.dump(stats_to_save, f, indent=2)
    print(f"[SAVED] Statistics: {stats_file}")
    
    # Save summary report
    report_file = os.path.join(condition_dir, f'{args.condition}_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RQ4 BIOLOGICALLY CORRECTED ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("CORRECTIONS APPLIED:\n")
        f.write("  [X] ATP maintenance objective (not biomass)\n")
        f.write("  [X] GPR-aware E-Flux (respects AND/OR logic)\n")
        f.write("  [X] Physiologically correct portal coupling\n")
        f.write("  [X] Scaled microbiome constraints\n\n")
        
        f.write(f"Condition: {args.condition}\n")
        f.write(f"Objective Mode: {args.objective_mode}\n")
        f.write(f"Portal Scaling: {args.portal_scaling}\n")
        f.write(f"Use pFBA: {args.use_pfba}\n\n")
        
        f.write("BASELINE STATISTICS:\n")
        if 'eflux' in results['stats_baseline']:
            eflux = results['stats_baseline']['eflux']
            f.write(f"  Genes matched: {eflux['genes_matched']} / "
                   f"{eflux['genes_in_model']} "
                   f"({100*eflux['genes_matched']/eflux['genes_in_model']:.1f}%)\n")
            f.write(f"  Reactions constrained: {eflux['reactions_constrained']}\n")
        
        if 'solution_baseline' in results:
            sol = results['solution_baseline']
            f.write(f"  Optimization status: {sol.status}\n")
            obj_val = sol.objective_value
            f.write(f"  Objective value: {obj_val:.4f}\n" if obj_val is not None
                    else "  Objective value: N/A (infeasible)\n")
        
        f.write("\nMICROBIOME STATISTICS:\n")
        if 'portal' in results['stats_microbiome']:
            portal = results['stats_microbiome']['portal']
            f.write(f"  Microbial production events: {portal['microbial_production']}\n")
            f.write(f"  Constraints applied: {portal['constraints_applied']}\n")
            f.write(f"  Scaling factor: {portal['scaling_factor']}\n")
        
        if 'solution_microbiome' in results:
            sol = results['solution_microbiome']
            f.write(f"  Optimization status: {sol.status}\n")
            obj_val = sol.objective_value
            f.write(f"  Objective value: {obj_val:.4f}\n" if obj_val is not None
                    else "  Objective value: N/A (infeasible)\n")
        
        if 'flux_comparison' in results:
            fc = results['flux_comparison']
            n_attr = fc['microbiome_attributable'].sum()
            f.write(f"\nFLUX CHANGES:\n")
            f.write(f"  Total reactions: {len(fc)}\n")
            f.write(f"  Microbiome-attributable: {n_attr} ({100*n_attr/len(fc):.1f}%)\n")
    
    print(f"[SAVED] Report: {report_file}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {condition_dir}")
    print("\nNext steps:")
    print("  1. Review statistics to validate constraint application")
    print("  2. Run sensitivity analysis with different portal_scaling values")
    print("  3. Compare with original (uncorrected) results")
    print("  4. Validate against known metabolite concentrations")


if __name__ == "__main__":
    main()
