#!/usr/bin/env python3
"""
RQ3: FINAL COMPLETE VERSION - REVISED
======================================
"""

import os
import sys
import json
import re
import argparse
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import cobra
from cobra.io import load_json_model
from cobra.flux_analysis import pfba, flux_variability_analysis

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# =============================================================================
# DIETARY CONSTRAINTS
# =============================================================================

def load_dietary_bounds(diet_bounds_file: str, condition_mapping: Optional[Dict]) -> Dict:
    """Load dietary bounds from JSON file with optional condition mapping."""
    with open(diet_bounds_file, 'r') as f:
        bounds_data = json.load(f)
    
    if condition_mapping:
        mapped_bounds = {}
        for user_condition, file_condition in condition_mapping.items():
            if file_condition in bounds_data:
                mapped_bounds[user_condition] = bounds_data[file_condition]
                print(f"[INFO] Mapped '{user_condition}' -> '{file_condition}'")
        return mapped_bounds
    return bounds_data


def apply_dietary_bounds(model: cobra.Model, diet_bounds: Dict) -> int:
    """Apply dietary bounds to model reactions."""
    n_applied = 0
    for rxn_id, bounds in diet_bounds.items():
        try:
            reaction = model.reactions.get_by_id(rxn_id)
            reaction.lower_bound = bounds[0]
            reaction.upper_bound = bounds[1]
            n_applied += 1
        except KeyError:
            pass
    return n_applied


# =============================================================================
# DATA LOADING
# =============================================================================

def load_single_cell_data(data_path: str, metadata_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load single-cell expression data and metadata."""
    expression_matrix = pd.read_csv(data_path, index_col=0)
    metadata = pd.read_csv(metadata_path, index_col=0)
    
    common_cells = metadata.index.intersection(expression_matrix.columns)
    metadata = metadata.loc[common_cells]
    expression_matrix = expression_matrix[common_cells]
    
    print(f"[INFO] Loaded {len(common_cells)} cells × {len(expression_matrix)} genes")
    return expression_matrix, metadata


def aggregate_expression(expression_matrix: pd.DataFrame, metadata: pd.DataFrame, 
                        cell_type_column: str = "cell_type", 
                        condition_column: str = "diet", 
                        min_cells: int = 10) -> Tuple[Dict, pd.DataFrame]:
    """
    Aggregate single-cell expression to pseudo-bulk profiles.
    
    NOTE ON VARIANCE LOSS (Feedback D):
    -----------------------------------
    This aggregation step creates a single mean expression profile per 
    cell type × condition combination. This means:
    - Cell-to-cell variance within each group is LOST
    - Downstream flux solutions are DETERMINISTIC (one solution per group)
    - Statistical comparisons are threshold-based, not inferential
    - Consider the aggregation_summary for cell count context
    """
    pseudo_bulk_dict = {}
    aggregation_records = []
    
    for cell_type in metadata[cell_type_column].unique():
        condition_profiles = {}
        
        for condition in metadata[condition_column].unique():
            cond_cell_ids = metadata[
                (metadata[cell_type_column] == cell_type) & 
                (metadata[condition_column] == condition)
            ].index.tolist()
            
            n_cells = len(cond_cell_ids)
            
            if n_cells < min_cells:
                print(f"[WARN] Skipping {cell_type} × {condition}: only {n_cells} cells")
                continue
            
            condition_expr = expression_matrix[cond_cell_ids]
            agg_profile = condition_expr.mean(axis=1)
            
            # Also compute variance statistics for documentation
            agg_std = condition_expr.std(axis=1)
            agg_cv = (agg_std / (agg_profile + 1e-10))  # coefficient of variation
            
            condition_profiles[condition] = agg_profile.copy()
            
            # Record aggregation stats including variance info
            aggregation_records.append({
                'cell_type': cell_type,
                'condition': condition,
                'n_cells': n_cells,
                'genes_detected': (agg_profile > 0).sum(),
                'mean_expression': agg_profile.mean(),
                'median_expression': agg_profile.median(),
                'max_expression': agg_profile.max(),
                'mean_cv': agg_cv[agg_profile > 0].mean(),  # mean CV of expressed genes
                'note': 'Cell-level variance lost in aggregation'
            })
        
        if len(condition_profiles) > 0:
            pseudo_bulk_dict[cell_type] = pd.DataFrame(condition_profiles).copy()
            print(f"[INFO] {cell_type}: {len(condition_profiles)} conditions")
    
    aggregation_summary = pd.DataFrame(aggregation_records)
    
    return pseudo_bulk_dict, aggregation_summary


# =============================================================================
# GPR LOGIC PARSING (Feedback B - REVISED)
# =============================================================================

def parse_gpr_expression(gpr_rule: str, gene_expression: Dict[str, float]) -> Optional[float]:
    """
    Parse GPR (Gene-Protein-Reaction) rule and compute expression score using
    proper boolean logic.
    
    BIOLOGICAL RATIONALE (Feedback B):
    ----------------------------------
    - AND (protein complexes): All subunits required -> use MINIMUM expression
      Example: "geneA and geneB" -> min(expr_A, expr_B)
      Rationale: Complex formation is limited by the least abundant subunit
    
    - OR (isozymes): Any isoform sufficient -> use MAXIMUM expression
      Example: "geneA or geneB" -> max(expr_A, expr_B)
      Rationale: Reaction can proceed via whichever enzyme is most available
    
    NOTE: This replaces the previous mean() heuristic which could:
    - Overestimate complex capacity (if one subunit is low)
    - Underestimate isozyme capacity (by averaging instead of taking max)
    
    Args:
        gpr_rule: Gene-protein-reaction rule string (e.g., "geneA and (geneB or geneC)")
        gene_expression: Dictionary mapping gene names/IDs to expression values
    
    Returns:
        Computed expression score, or None if no genes found
    """
    if not gpr_rule or gpr_rule.strip() == '':
        return None
    
    # Normalize the GPR rule
    rule = gpr_rule.strip()
    
    # Handle simple single-gene case
    if ' and ' not in rule.lower() and ' or ' not in rule.lower():
        # Single gene or parenthesized single gene
        gene = rule.strip('() ')
        return gene_expression.get(gene, None)
    
    try:
        return _evaluate_gpr_recursive(rule, gene_expression)
    except Exception as e:
        # Fallback to simple extraction if parsing fails
        return _fallback_gpr_evaluation(rule, gene_expression)


def _evaluate_gpr_recursive(rule: str, gene_expression: Dict[str, float]) -> Optional[float]:
    """
    Recursively evaluate GPR expression with proper operator precedence.
    
    Handles nested parentheses and mixed AND/OR operations.
    """
    rule = rule.strip()
    
    # Remove outer parentheses if they wrap the entire expression
    while rule.startswith('(') and rule.endswith(')'):
        # Check if these parentheses actually match
        depth = 0
        matched = True
        for i, char in enumerate(rule[:-1]):
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            if depth == 0 and i < len(rule) - 2:
                matched = False
                break
        if matched:
            rule = rule[1:-1].strip()
        else:
            break
    
    # Find the main operator (lowest precedence, outside parentheses)
    # OR has lower precedence than AND
    or_pos = _find_operator_position(rule, ' or ')
    if or_pos != -1:
        left = rule[:or_pos].strip()
        right = rule[or_pos + 4:].strip()
        left_val = _evaluate_gpr_recursive(left, gene_expression)
        right_val = _evaluate_gpr_recursive(right, gene_expression)
        
        # OR logic: MAXIMUM (any isoform sufficient)
        if left_val is not None and right_val is not None:
            return max(left_val, right_val)
        elif left_val is not None:
            return left_val
        elif right_val is not None:
            return right_val
        return None
    
    and_pos = _find_operator_position(rule, ' and ')
    if and_pos != -1:
        left = rule[:and_pos].strip()
        right = rule[and_pos + 5:].strip()
        left_val = _evaluate_gpr_recursive(left, gene_expression)
        right_val = _evaluate_gpr_recursive(right, gene_expression)
        
        # AND logic: MINIMUM (all subunits required)
        if left_val is not None and right_val is not None:
            return min(left_val, right_val)
        # If one part is missing, we cannot compute the complex
        return None
    
    # Base case: single gene
    gene = rule.strip('() ')
    return gene_expression.get(gene, None)


def _find_operator_position(rule: str, operator: str) -> int:
    """Find the position of an operator that is not inside parentheses."""
    depth = 0
    op_lower = operator.lower()
    rule_lower = rule.lower()
    
    i = 0
    while i < len(rule_lower):
        if rule[i] == '(':
            depth += 1
        elif rule[i] == ')':
            depth -= 1
        elif depth == 0 and rule_lower[i:i+len(op_lower)] == op_lower:
            return i
        i += 1
    return -1


def _fallback_gpr_evaluation(rule: str, gene_expression: Dict[str, float]) -> Optional[float]:
    """
    Fallback GPR evaluation when parsing fails.
    Extracts all genes and applies conservative logic.
    """
    # Extract all gene identifiers
    gene_pattern = r'[A-Za-z][A-Za-z0-9_\-\.]*'
    potential_genes = re.findall(gene_pattern, rule)
    
    # Filter out operators
    operators = {'and', 'or', 'AND', 'OR', 'And', 'Or'}
    genes = [g for g in potential_genes if g not in operators]
    
    # Get expression values
    values = [gene_expression[g] for g in genes if g in gene_expression and gene_expression[g] > 0]
    
    if not values:
        return None
    
    # Conservative: if rule contains 'and', use min; otherwise use max
    if ' and ' in rule.lower():
        return min(values)
    else:
        return max(values)


# =============================================================================
# QUANTILE NORMALIZATION (Feedback C - REVISED)
# =============================================================================

def compute_quantile_references(pseudo_bulk_dict: Dict, quantile: float = 0.95,
                               strategy: str = 'per_condition') -> Dict:
    """
    Compute quantile reference values for E-Flux normalization.
    
    NORMALIZATION STRATEGIES (Feedback C):
    --------------------------------------
    Different strategies can significantly affect results:
    
    - 'global': Single quantile across ALL samples
      Pros: Simple, consistent baseline
      Cons: May obscure differential expression magnitudes between conditions
    
    - 'per_condition': Separate quantile per dietary condition
      Pros: Preserves relative differences between cell types within condition
      Cons: May not be directly comparable across conditions
    
    - 'per_celltype': Separate quantile per cell type
      Pros: Accounts for cell-type-specific expression ranges
      Cons: Loses ability to compare absolute levels across cell types
    
    - 'per_sample': Separate quantile per cell type × condition
      Pros: Each sample normalized independently
      Cons: Most aggressive normalization, may over-normalize
    
    Args:
        pseudo_bulk_dict: Dictionary of pseudo-bulk expression DataFrames
        quantile: Quantile value (default 0.95)
        strategy: One of 'global', 'per_condition', 'per_celltype', 'per_sample'
    
    Returns:
        Dictionary of quantile reference values keyed appropriately
    """
    print(f"\n[QUANTILE] Computing {quantile*100:.0f}th percentile references...")
    print(f"[QUANTILE] Strategy: {strategy}")
    
    quantile_refs = {}
    
    if strategy == 'global':
        # Single global quantile (original behavior)
        all_values = []
        for cell_type, expr_df in pseudo_bulk_dict.items():
            for condition in expr_df.columns:
                values = expr_df[condition].values
                all_values.extend(values[values > 0])
        
        global_q = np.quantile(all_values, quantile)
        quantile_refs['global'] = global_q
        print(f"[QUANTILE] Global reference: {global_q:.4f}")
        
    elif strategy == 'per_condition':
        # Separate quantile per condition
        condition_values = defaultdict(list)
        for cell_type, expr_df in pseudo_bulk_dict.items():
            for condition in expr_df.columns:
                values = expr_df[condition].values
                condition_values[condition].extend(values[values > 0])
        
        for condition, values in condition_values.items():
            q = np.quantile(values, quantile)
            quantile_refs[condition] = q
            print(f"[QUANTILE] {condition}: {q:.4f}")
            
    elif strategy == 'per_celltype':
        # Separate quantile per cell type
        for cell_type, expr_df in pseudo_bulk_dict.items():
            all_values = []
            for condition in expr_df.columns:
                values = expr_df[condition].values
                all_values.extend(values[values > 0])
            
            q = np.quantile(all_values, quantile)
            quantile_refs[cell_type] = q
            print(f"[QUANTILE] {cell_type}: {q:.4f}")
            
    elif strategy == 'per_sample':
        # Separate quantile per cell type × condition
        for cell_type, expr_df in pseudo_bulk_dict.items():
            for condition in expr_df.columns:
                values = expr_df[condition].values
                values = values[values > 0]
                if len(values) > 0:
                    q = np.quantile(values, quantile)
                else:
                    q = 1.0  # fallback
                quantile_refs[(cell_type, condition)] = q
        print(f"[QUANTILE] Computed {len(quantile_refs)} sample-specific references")
        
    else:
        raise ValueError(f"Unknown normalization strategy: {strategy}")
    
    return quantile_refs


def get_quantile_reference(quantile_refs: Dict, cell_type: str, condition: str, 
                          strategy: str) -> float:
    """Get the appropriate quantile reference for a given sample."""
    if strategy == 'global':
        return quantile_refs['global']
    elif strategy == 'per_condition':
        return quantile_refs.get(condition, quantile_refs.get(list(quantile_refs.keys())[0]))
    elif strategy == 'per_celltype':
        return quantile_refs.get(cell_type, quantile_refs.get(list(quantile_refs.keys())[0]))
    elif strategy == 'per_sample':
        return quantile_refs.get((cell_type, condition), 1.0)
    else:
        return 1.0


# =============================================================================
# E-FLUX IMPLEMENTATION (REVISED)
# =============================================================================

def apply_eflux_correctly(model: cobra.Model, pseudo_bulk_dict: Dict, 
                         diet_bounds_dict: Optional[Dict],
                         eflux_quantile: float = 0.95,
                         eflux_floor: float = 0.1, 
                         eflux_cap: float = 1000.0,
                         objective_reaction: str = "BIOMASS_mm_1_no_glygln",
                         normalization_strategy: str = 'per_condition',
                         run_fva: bool = False,
                         fva_fraction: float = 0.9) -> Tuple[Dict, Optional[Dict]]:
    """
    Apply E-Flux with CORRECT implementation using CONSTRAINING.
    
    REVISIONS APPLIED:
    ------------------
    - Uses proper GPR boolean logic (Feedback B)
    - Flexible quantile normalization (Feedback C)
    - Optional FVA for uncertainty bounds (Feedback D)
    
    Args:
        model: COBRApy model
        pseudo_bulk_dict: Pseudo-bulk expression profiles
        diet_bounds_dict: Dietary constraint bounds
        eflux_quantile: Quantile for normalization
        eflux_floor: Minimum normalized expression value
        eflux_cap: Maximum normalized expression value
        objective_reaction: Objective function reaction ID
        normalization_strategy: 'global', 'per_condition', 'per_celltype', 'per_sample'
        run_fva: Whether to run FVA for uncertainty bounds
        fva_fraction: Fraction of optimal objective for FVA
    
    Returns:
        solutions_dict: Dictionary of pFBA solutions
        fva_dict: Dictionary of FVA results (if run_fva=True)
    """
    print("\n" + "="*80)
    print("E-FLUX IMPLEMENTATION (REVISED)")
    print("="*80)
    print("[FIX] [OK] Using CONSTRAINING (min/max) not SCALING!")
    print(f"[FIX] [OK] Using proper GPR boolean logic (AND=min, OR=max)")
    print(f"[FIX] [OK] Normalization strategy: {normalization_strategy}")
    if run_fva:
        print(f"[FIX] [OK] Running FVA for uncertainty bounds (fraction={fva_fraction})")
    print("="*80 + "\n")
    
    model.objective = objective_reaction
    
    # Build gene name mapping
    model_gene_name_to_id = {}
    model_gene_id_to_name = {}
    for gene in model.genes:
        gene_name = gene.name if hasattr(gene, 'name') and gene.name else gene.id
        model_gene_name_to_id[gene_name] = gene.id
        model_gene_id_to_name[gene.id] = gene_name
    
    # Compute quantile references
    quantile_refs = compute_quantile_references(
        pseudo_bulk_dict, eflux_quantile, normalization_strategy
    )
    
    # Apply E-Flux
    print("\n[STEP] Applying E-Flux with CONSTRAINING...")
    
    solutions_dict = {}
    fva_dict = {} if run_fva else None
    gpr_stats = {'and_rules': 0, 'or_rules': 0, 'simple_rules': 0, 'complex_rules': 0}
    
    for cell_type, expression_df in pseudo_bulk_dict.items():
        print(f"\n[INFO] === {cell_type} ===")
        
        cell_type_solutions = {}
        cell_type_fva = {} if run_fva else None
        expression_genes = set(expression_df.index)
        matched = expression_genes & set(model_gene_name_to_id.keys())
        print(f"[INFO] Matched {len(matched)} genes to model")
        
        for condition in expression_df.columns:
            print(f"  [INFO] Processing {condition}...")
            
            model_copy = model.copy()
            model_copy.objective = objective_reaction
            
            # Apply dietary bounds
            if diet_bounds_dict and condition in diet_bounds_dict:
                n_applied = apply_dietary_bounds(model_copy, diet_bounds_dict[condition])
                print(f"    [DIET] Applied {n_applied} bounds")
            
            # Get appropriate quantile reference
            q_ref = get_quantile_reference(quantile_refs, cell_type, condition, 
                                          normalization_strategy)
            
            # Normalize expression
            expression_profile = expression_df[condition].copy()
            normalized_expr = (expression_profile / q_ref).clip(lower=eflux_floor, upper=eflux_cap)
            
            # Build gene expression dictionary (handle both names and IDs)
            gene_expr_dict = {}
            for gene_name in normalized_expr.index:
                if normalized_expr[gene_name] > 0:
                    gene_expr_dict[gene_name] = normalized_expr[gene_name]
                    # Also add by ID if we have a mapping
                    if gene_name in model_gene_name_to_id:
                        gene_expr_dict[model_gene_name_to_id[gene_name]] = normalized_expr[gene_name]
            
            print(f"    [DEBUG] Normalized mean: {normalized_expr.mean():.4f}, "
                  f"quantile ref: {q_ref:.4f}")
            
            # CRITICAL: Apply E-Flux using proper GPR logic
            reactions_constrained = 0
            for reaction in model_copy.reactions:
                if not reaction.gene_reaction_rule:
                    continue
                
                # Use proper GPR evaluation (Feedback B)
                gpr_rule = reaction.gene_reaction_rule
                
                # Track GPR complexity for stats
                if ' and ' in gpr_rule.lower() and ' or ' in gpr_rule.lower():
                    gpr_stats['complex_rules'] += 1
                elif ' and ' in gpr_rule.lower():
                    gpr_stats['and_rules'] += 1
                elif ' or ' in gpr_rule.lower():
                    gpr_stats['or_rules'] += 1
                else:
                    gpr_stats['simple_rules'] += 1
                
                expr_score = parse_gpr_expression(gpr_rule, gene_expr_dict)
                
                if expr_score is not None and expr_score > 0:
                    val = float(expr_score)
                    
                    # CONSTRAIN bounds
                    old_lb, old_ub = reaction.lower_bound, reaction.upper_bound
                    
                    if reaction.lower_bound < 0:
                        reaction.lower_bound = max(reaction.lower_bound, -val)
                    
                    reaction.upper_bound = min(reaction.upper_bound, val)
                    
                    if (reaction.lower_bound != old_lb) or (reaction.upper_bound != old_ub):
                        reactions_constrained += 1
            
            print(f"    [EFLUX] Constrained {reactions_constrained} reactions")
            
            # Run pFBA
            try:
                print(f"    [PFBA] Running parsimonious FBA...")
                pfba_solution = pfba(model_copy)
                
                if pfba_solution.status == 'optimal':
                    obj_val = pfba_solution.objective_value
                    flux_vals = pfba_solution.fluxes.values
                    n_nonzero = np.sum(np.abs(flux_vals) > 1e-6)
                    total_flux = np.sum(np.abs(flux_vals))
                    
                    print(f"    [RESULT] obj={obj_val:.6f}, nonzero={n_nonzero}, "
                          f"total_flux={total_flux:.2f}")
                    cell_type_solutions[condition] = pfba_solution
                    
                    # Optional FVA for uncertainty bounds (Feedback D)
                    if run_fva:
                        print(f"    [FVA] Computing flux variability...")
                        try:
                            fva_result = flux_variability_analysis(
                                model_copy, 
                                fraction_of_optimum=fva_fraction,
                                loopless=False
                            )
                            cell_type_fva[condition] = fva_result
                            
                            # Report FVA summary
                            ranges = fva_result['maximum'] - fva_result['minimum']
                            n_variable = np.sum(ranges > 1e-6)
                            print(f"    [FVA] {n_variable} reactions with flux variability")
                        except Exception as e:
                            print(f"    [FVA WARN] {e}")
                            cell_type_fva[condition] = None
                else:
                    print(f"    [WARN] Status: {pfba_solution.status}")
                    cell_type_solutions[condition] = pfba_solution
                    
            except Exception as e:
                print(f"    [ERROR] {e}")
                cell_type_solutions[condition] = None
        
        solutions_dict[cell_type] = cell_type_solutions
        if run_fva:
            fva_dict[cell_type] = cell_type_fva
    
    # Report GPR stats (first cell type only to avoid repetition)
    print(f"\n[GPR STATS] Rule types encountered:")
    print(f"  - Simple (single gene): {gpr_stats['simple_rules']}")
    print(f"  - AND only (complexes): {gpr_stats['and_rules']}")
    print(f"  - OR only (isozymes): {gpr_stats['or_rules']}")
    print(f"  - Complex (mixed AND/OR): {gpr_stats['complex_rules']}")
    
    # Validation checks
    _run_validation_checks(solutions_dict)
    
    return solutions_dict, fva_dict


def _run_validation_checks(solutions_dict: Dict) -> None:
    """Run validation checks on E-Flux solutions."""
    print("\n" + "="*80)
    print("VALIDATION: Cell Type Diversity Check")
    print("="*80)
    
    cell_types = list(solutions_dict.keys())
    if len(cell_types) >= 2:
        ct1, ct2 = cell_types[0], cell_types[1]
        conditions = list(solutions_dict[ct1].keys())
        
        if len(conditions) > 0:
            cond = conditions[0]
            s1, s2 = solutions_dict[ct1].get(cond), solutions_dict[ct2].get(cond)
            
            if s1 and s2 and s1.status == 'optimal' and s2.status == 'optimal':
                f1, f2 = s1.fluxes.values, s2.fluxes.values
                corr = np.corrcoef(f1, f2)[0, 1]
                n_diff = np.sum(np.abs(f1 - f2) > 0.01)
                
                print(f"[VALIDATION] {ct1} vs {ct2} ({cond}):")
                print(f"  Correlation: {corr:.6f}")
                print(f"  Different reactions: {n_diff}")
                
                if corr < 0.99 and n_diff > 100:
                    print("  [OK] SUCCESS: Cell types produce DIFFERENT solutions!")
                else:
                    print("  [!] Still similar - may be biological")
    
    # Check dietary differences
    print("\n[VALIDATION] Dietary Response Check:")
    for cell_type, cond_dict in list(solutions_dict.items())[:5]:
        conditions = list(cond_dict.keys())
        if len(conditions) >= 2:
            c1, c2 = conditions[0], conditions[1]
            s1, s2 = cond_dict.get(c1), cond_dict.get(c2)
            
            if s1 and s2 and s1.status == 'optimal' and s2.status == 'optimal':
                f1, f2 = s1.fluxes.values, s2.fluxes.values
                corr = np.corrcoef(f1, f2)[0, 1]
                n_diff = np.sum(np.abs(f1 - f2) > 0.01)
                
                status = "[OK] GOOD" if corr < 0.95 and n_diff > 100 else "[!] SIMILAR"
                print(f"[{status}] {cell_type}: {c1} vs {c2}, Corr={corr:.4f}, Different={n_diff}")
    
    print("="*80 + "\n")


# =============================================================================
# STATISTICAL TESTING (Feedback A - REVISED)
# =============================================================================

def perform_statistical_tests(solutions_dict: Dict, model: cobra.Model, 
                             baseline_condition: str = "Chow",
                             test_conditions: Optional[List[str]] = None, 
                             fold_change_threshold: float = 1.5,
                             abs_change_threshold: float = 0.1, 
                             results_dir: str = "results",
                             fva_dict: Optional[Dict] = None) -> pd.DataFrame:
    """
    Perform statistical testing on flux solutions.
    
    IMPORTANT METHODOLOGICAL NOTES (Feedback A & D):
    ------------------------------------------------
    1. DETERMINISTIC SOLUTIONS: Each cell type × condition has ONE flux solution
       from pFBA. There is no variance from repeated sampling or cell-level data.
    
    2. RELATIVE MAGNITUDE (formerly "Cohen's D"): The metric computed here is:
       
           relative_magnitude = flux_change / |baseline_flux|
       
       This is a RELATIVE DIFFERENCE or signed percent change, NOT Cohen's d.
       Cohen's d requires population variance: d = (μ1 - μ2) / σ_pooled
       Since we have single deterministic solutions, variance is unavailable.
    
    3. THRESHOLD-BASED SIGNIFICANCE: Without variance, we cannot compute p-values.
       "Significance" here means the change exceeds user-defined thresholds:
       - Fold change threshold (default 1.5x)
       - Absolute change threshold (default 0.1)
       - Relative magnitude threshold (default 0.5)
    
    4. INTERPRETATION: These are descriptive differences between model predictions,
       not statistically inferred differences with confidence intervals.
    """
    print("\n" + "="*80)
    print("STATISTICAL TESTING (REVISED)")
    print("="*80)
    print("[NOTE] Solutions are DETERMINISTIC - no variance estimates available")
    print("[NOTE] 'Significance' is threshold-based, not inferential")
    print("[NOTE] 'Relative Magnitude' replaces incorrectly named 'Cohen's D'")
    print("="*80 + "\n")
    
    if test_conditions is None:
        all_conditions = set()
        for ct_dict in solutions_dict.values():
            all_conditions.update(ct_dict.keys())
        test_conditions = [c for c in all_conditions if c != baseline_condition]
    
    # Collect flux data
    flux_data = defaultdict(lambda: defaultdict(dict))
    for cell_type, cond_dict in solutions_dict.items():
        for condition, solution in cond_dict.items():
            if solution and solution.status == 'optimal':
                for rxn_id, flux_val in solution.fluxes.items():
                    flux_data[rxn_id][cell_type][condition] = flux_val
    
    # Collect FVA data if available
    fva_ranges = defaultdict(lambda: defaultdict(dict))
    if fva_dict:
        for cell_type, cond_fva in fva_dict.items():
            for condition, fva_result in cond_fva.items():
                if fva_result is not None:
                    for rxn_id in fva_result.index:
                        fva_ranges[rxn_id][cell_type][condition] = {
                            'min': fva_result.loc[rxn_id, 'minimum'],
                            'max': fva_result.loc[rxn_id, 'maximum']
                        }
    
    stats_records = []
    
    for rxn_id in flux_data.keys():
        rxn = model.reactions.get_by_id(rxn_id)
        
        for test_condition in test_conditions:
            for cell_type in flux_data[rxn_id].keys():
                baseline_flux = flux_data[rxn_id][cell_type].get(baseline_condition, 0.0)
                test_flux = flux_data[rxn_id][cell_type].get(test_condition, None)
                
                if test_flux is None:
                    continue
                
                flux_change = test_flux - baseline_flux
                abs_flux_change = abs(flux_change)
                
                # Compute fold change
                if abs(baseline_flux) > 1e-10:
                    fold_change = test_flux / baseline_flux
                elif abs(test_flux) > 1e-10:
                    fold_change = np.inf if test_flux > 0 else -np.inf
                else:
                    fold_change = 1.0
                
                # REVISED: Relative Magnitude (NOT Cohen's d) - Feedback A
                # This is the signed relative difference, not a standardized effect size
                if abs(baseline_flux) > 1e-10:
                    relative_magnitude = flux_change / abs(baseline_flux)
                elif abs(test_flux) > 1e-10:
                    relative_magnitude = np.sign(test_flux) * 100.0  # Large relative change
                else:
                    relative_magnitude = 0.0
                
                abs_fold_change = abs(fold_change) if np.isfinite(fold_change) else 999.0
                log2fc = np.log2(abs_fold_change + 1e-10)
                
                record = {
                    'reaction_id': rxn_id,
                    'reaction_name': rxn.name,
                    'subsystem': rxn.subsystem if rxn.subsystem else "Unknown",
                    'cell_type': cell_type,
                    'comparison': f"{test_condition}_vs_{baseline_condition}",
                    'baseline_flux': baseline_flux,
                    'test_flux': test_flux,
                    'flux_change': flux_change,
                    'abs_flux_change': abs_flux_change,
                    'fold_change': fold_change,
                    'abs_fold_change': abs_fold_change,
                    'log2_fold_change': log2fc,
                    # RENAMED from cohens_d (Feedback A)
                    'relative_magnitude': relative_magnitude,
                }
                
                # Add FVA ranges if available (Feedback D)
                if fva_dict:
                    baseline_fva = fva_ranges[rxn_id][cell_type].get(baseline_condition, {})
                    test_fva = fva_ranges[rxn_id][cell_type].get(test_condition, {})
                    
                    if baseline_fva:
                        record['baseline_flux_min'] = baseline_fva.get('min', np.nan)
                        record['baseline_flux_max'] = baseline_fva.get('max', np.nan)
                    if test_fva:
                        record['test_flux_min'] = test_fva.get('min', np.nan)
                        record['test_flux_max'] = test_fva.get('max', np.nan)
                        
                        # Check if ranges overlap (conservative significance)
                        if baseline_fva and test_fva:
                            ranges_overlap = (
                                baseline_fva.get('max', np.inf) >= test_fva.get('min', -np.inf) and
                                test_fva.get('max', np.inf) >= baseline_fva.get('min', -np.inf)
                            )
                            record['fva_ranges_overlap'] = ranges_overlap
                
                stats_records.append(record)
    
    stats_df = pd.DataFrame(stats_records)
    
    if len(stats_df) == 0:
        print("[WARN] No statistical records generated")
        return stats_df
    
    # Apply threshold-based criteria
    for comparison in stats_df['comparison'].unique():
        mask = stats_df['comparison'] == comparison
        
        meets_fc = stats_df.loc[mask, 'abs_fold_change'] > fold_change_threshold
        meets_abs = stats_df.loc[mask, 'abs_flux_change'] > abs_change_threshold
        # RENAMED: uses relative_magnitude instead of cohens_d
        meets_effect = np.abs(stats_df.loc[mask, 'relative_magnitude']) > 0.5
        
        stats_df.loc[mask, 'meets_fc_threshold'] = meets_fc
        stats_df.loc[mask, 'meets_abs_threshold'] = meets_abs
        stats_df.loc[mask, 'meets_effect_threshold'] = meets_effect
        
        n_criteria = meets_fc.astype(int) + meets_abs.astype(int) + meets_effect.astype(int)
        stats_df.loc[mask, 'significant'] = n_criteria >= 2
        stats_df.loc[mask, 'significant_strict'] = n_criteria == 3
    
    # Save results
    os.makedirs(os.path.join(results_dir, "statistics"), exist_ok=True)
    output_path = os.path.join(results_dir, "statistics", "statistical_tests.csv")
    stats_df.to_csv(output_path, index=False)
    
    # Save methodology notes
    _save_methodology_notes(results_dir)
    
    # Print summary
    n_sig = stats_df['significant'].sum()
    total = len(stats_df)
    print(f"\n[SUMMARY] Threshold-significant reactions: {n_sig}/{total} ({100*n_sig/total:.1f}%)")
    
    for ct in stats_df['cell_type'].unique():
        ct_mask = stats_df['cell_type'] == ct
        n_sig_ct = stats_df.loc[ct_mask, 'significant'].sum()
        total_ct = ct_mask.sum()
        print(f"  {ct}: {n_sig_ct}/{total_ct} ({100*n_sig_ct/total_ct:.1f}%)")
    
    return stats_df


def _save_methodology_notes(results_dir: str) -> None:
    """Save methodology notes to accompany results."""
    notes = """
METHODOLOGY NOTES FOR STATISTICAL ANALYSIS
==========================================

1. DETERMINISTIC SOLUTIONS
--------------------------
Each cell type x dietary condition combination produces a SINGLE flux solution
via parsimonious FBA (pFBA). This is a deterministic optimization - running
the same model with the same constraints will always yield the same result.

Implication: There is no variance from sampling or biological replicates at
the flux level. Any apparent "variation" comes from comparing different
cell types or conditions.


2. RELATIVE MAGNITUDE METRIC (formerly "Cohen's D")
--------------------------------------------------
The "relative_magnitude" column is computed as:

    relative_magnitude = (test_flux - baseline_flux) / |baseline_flux|

This is a RELATIVE DIFFERENCE or signed percent change. It is NOT Cohen's d.

Cohen's d is a standardized effect size that requires variance:
    d = (mean1 - mean2) / sigma_pooled

Since pseudo-bulk aggregation produces single values (not distributions),
we cannot compute a true standardized effect size.

Interpretation of relative_magnitude:
- 1.0 means the flux doubled relative to baseline
- -0.5 means the flux decreased by 50%
- Values > 100 indicate the baseline was near zero


3. THRESHOLD-BASED "SIGNIFICANCE"
---------------------------------
Without variance estimates, we cannot compute p-values or confidence intervals.
"Significance" in this analysis means the change exceeds user-defined thresholds:

- Fold change > 1.5 (or as specified)
- Absolute change > 0.1 (or as specified)
- Relative magnitude > 0.5

A reaction is marked "significant" if it meets >=2 of these criteria.
A reaction is marked "significant_strict" if it meets all 3 criteria.

CAUTION: These are descriptive thresholds, not statistical significance.


4. FLUX VARIABILITY ANALYSIS (FVA)
----------------------------------
If FVA was run, additional columns provide:
- baseline_flux_min/max: Range of feasible fluxes at baseline
- test_flux_min/max: Range of feasible fluxes at test condition
- fva_ranges_overlap: Whether the feasible ranges overlap

Non-overlapping FVA ranges provide stronger evidence that the flux truly
differs between conditions, as the flux MUST differ to achieve near-optimal
objective values.


5. GPR LOGIC
------------
Gene expression is mapped to reactions using boolean logic:
- AND rules (protein complexes): MINIMUM expression
- OR rules (isozymes): MAXIMUM expression

This reflects biological constraints where:
- Complexes require all subunits (limited by least abundant)
- Isozymes can substitute (any one is sufficient)
"""
    
    # Use UTF-8 encoding explicitly for cross-platform compatibility
    with open(os.path.join(results_dir, "statistics", "METHODOLOGY_NOTES.txt"), 'w', encoding='utf-8') as f:
        f.write(notes)
    print(f"[INFO] Methodology notes saved to {results_dir}/statistics/METHODOLOGY_NOTES.txt")


# =============================================================================
# FLUX COMPARISON
# =============================================================================

def compare_fluxes(solutions_dict: Dict, model: cobra.Model, results_dir: str) -> pd.DataFrame:
    """Compare fluxes across conditions and cell types."""
    flux_data = defaultdict(lambda: defaultdict(dict))
    for ct, cond_dict in solutions_dict.items():
        for cond, sol in cond_dict.items():
            if sol and sol.status == 'optimal':
                for rxn_id, flux_val in sol.fluxes.items():
                    flux_data[rxn_id][ct][cond] = flux_val
    
    records = []
    for rxn_id in flux_data.keys():
        rxn = model.reactions.get_by_id(rxn_id)
        record = {
            'reaction_id': rxn_id,
            'reaction_name': rxn.name,
            'subsystem': rxn.subsystem or "Unknown"
        }
        
        for ct in flux_data[rxn_id].keys():
            for cond in flux_data[rxn_id][ct].keys():
                record[f'{ct}_{cond}'] = flux_data[rxn_id][ct][cond]
        
        records.append(record)
    
    df = pd.DataFrame(records)
    os.makedirs(os.path.join(results_dir, "statistics"), exist_ok=True)
    df.to_csv(os.path.join(results_dir, "statistics", "flux_comparison.csv"), index=False)
    return df


# =============================================================================
# VISUALIZATIONS
# =============================================================================

def create_visualizations(stats_df: pd.DataFrame, solutions_dict: Dict, 
                         results_dir: str, fva_dict: Optional[Dict] = None) -> None:
    """Create comprehensive visualizations."""
    print("\n[INFO] Creating visualizations...")
    
    viz_dir = os.path.join(results_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # 1. Cell-type response rates
    print("  - Response rates barplot...")
    ct_summary = stats_df.groupby('cell_type')['significant'].agg(['sum', 'count'])
    ct_summary['pct'] = 100 * ct_summary['sum'] / ct_summary['count']
    ct_summary = ct_summary.sort_values('pct', ascending=False)
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(ct_summary)), ct_summary['pct'])
    plt.xticks(range(len(ct_summary)), ct_summary.index, rotation=45, ha='right')
    plt.ylabel('% Reactions Exceeding Thresholds')
    plt.title('Cell-Type-Specific Metabolic Response Rates\n(Threshold-based, not statistical significance)')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'response_rates_by_celltype.png'), dpi=300)
    plt.close()
    
    # 2. Heatmap of significant reactions per cell type
    print("  - Response heatmap...")
    sig_counts = stats_df[stats_df['significant']].groupby(['cell_type', 'subsystem']).size().unstack(fill_value=0)
    if len(sig_counts) > 0:
        top_subsystems = sig_counts.sum(axis=0).nlargest(20).index
        sig_counts_top = sig_counts[top_subsystems]
        
        plt.figure(figsize=(14, 8))
        sns.heatmap(sig_counts_top.T, cmap='YlOrRd', annot=False, fmt='d', 
                   cbar_kws={'label': 'Reactions Exceeding Thresholds'})
        plt.xlabel('Cell Type')
        plt.ylabel('Subsystem')
        plt.title('Metabolic Changes by Cell Type and Subsystem\n(Count of reactions exceeding thresholds)')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'heatmap_responses.png'), dpi=300)
        plt.close()
    
    # 3. Objective values comparison
    print("  - Objective values plot...")
    obj_data = []
    for ct, cond_dict in solutions_dict.items():
        for cond, sol in cond_dict.items():
            if sol and sol.status == 'optimal':
                obj_data.append({
                    'cell_type': ct,
                    'condition': cond,
                    'objective': sol.objective_value
                })
    
    obj_df = pd.DataFrame(obj_data)
    if len(obj_df) > 0:
        plt.figure(figsize=(14, 6))
        for cond in obj_df['condition'].unique():
            cond_data = obj_df[obj_df['condition'] == cond].sort_values('objective', ascending=False)
            plt.scatter(range(len(cond_data)), cond_data['objective'], label=cond, s=100, alpha=0.7)
        
        plt.xlabel('Cell Type (sorted by objective)')
        plt.ylabel('Objective Value (Biomass)')
        plt.title('Predicted Metabolic Capacity Across Cell Types\n(Single deterministic solution per condition)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'objective_values.png'), dpi=300)
        plt.close()
    
    # 4. Top changed reactions - using relative_magnitude
    print("  - Top changed reactions...")
    top_changed = stats_df[stats_df['significant']].nlargest(30, 'abs_flux_change')
    
    if len(top_changed) > 0:
        plt.figure(figsize=(10, 8))
        y_pos = np.arange(len(top_changed))
        colors = ['red' if x < 0 else 'blue' for x in top_changed['flux_change']]
        plt.barh(y_pos, top_changed['flux_change'], color=colors, alpha=0.6)
        plt.yticks(y_pos, [f"{r[:30]}..." if len(r) > 30 else r for r in top_changed['reaction_name']])
        plt.xlabel('Flux Change (test - baseline)')
        plt.title('Top 30 Most Changed Reactions\n(Blue = increased, Red = decreased)')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'top_changed_reactions.png'), dpi=300)
        plt.close()
    
    # 5. NEW: Relative magnitude distribution
    print("  - Relative magnitude distribution...")
    plt.figure(figsize=(10, 6))
    sig_rel_mag = stats_df[stats_df['significant']]['relative_magnitude']
    sig_rel_mag_clipped = sig_rel_mag.clip(-10, 10)  # Clip for visualization
    plt.hist(sig_rel_mag_clipped, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Relative Magnitude of Change\n(flux_change / |baseline_flux|, clipped to ±10)')
    plt.ylabel('Count')
    plt.title('Distribution of Relative Magnitude\n(Threshold-significant reactions only)')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'relative_magnitude_distribution.png'), dpi=300)
    plt.close()
    
    print(f"  [OK] Visualizations saved to {viz_dir}/")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="RQ3: Final Complete (REVISED)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
REVISION NOTES:
  A. 'Cohen's D' renamed to 'Relative Magnitude' (not a true effect size)
  B. GPR logic uses MIN for AND, MAX for OR (proper boolean)
  C. Flexible quantile normalization strategies available
  D. Optional FVA for uncertainty bounds; methodology notes included
        """
    )
    
    # Required arguments
    parser.add_argument('--sc_data', required=True, help='Single-cell expression matrix')
    parser.add_argument('--sc_metadata', required=True, help='Cell metadata file')
    parser.add_argument('--diet_bounds_file', required=True, help='Dietary bounds JSON')
    parser.add_argument('--condition_mapping', required=True, help='JSON string mapping conditions')
    
    # Model parameters
    parser.add_argument('--model', default='iMM1415.json', help='Metabolic model file')
    parser.add_argument('--objective', default='BIOMASS_mm_1_no_glygln', help='Objective reaction')
    
    # E-Flux parameters
    parser.add_argument('--eflux_quantile', type=float, default=0.99, 
                       help='Quantile for normalization')
    parser.add_argument('--eflux_floor', type=float, default=0.001, 
                       help='Minimum normalized expression')
    parser.add_argument('--eflux_cap', type=float, default=10000, 
                       help='Maximum normalized expression')
    
    # NEW: Normalization strategy (Feedback C)
    parser.add_argument('--normalization_strategy', default='per_condition',
                       choices=['global', 'per_condition', 'per_celltype', 'per_sample'],
                       help='Quantile normalization strategy')
    
    # Statistical parameters
    parser.add_argument('--baseline', default='Chow', help='Baseline condition')
    parser.add_argument('--test_conditions', default=None, 
                       help='Comma-separated test conditions')
    parser.add_argument('--fold_change_threshold', type=float, default=1.5,
                       help='Fold change threshold for significance')
    parser.add_argument('--abs_change_threshold', type=float, default=0.1,
                       help='Absolute change threshold for significance')
    
    # NEW: FVA option (Feedback D)
    parser.add_argument('--run_fva', action='store_true',
                       help='Run FVA for uncertainty bounds')
    parser.add_argument('--fva_fraction', type=float, default=0.9,
                       help='Fraction of optimal for FVA')
    
    # Other parameters
    parser.add_argument('--min_cells', type=int, default=20, 
                       help='Minimum cells for aggregation')
    parser.add_argument('--results_dir', default='results_rq3_final_revised',
                       help='Output directory')
    
    args = parser.parse_args()
    
    condition_mapping = json.loads(args.condition_mapping)
    test_conditions = [c.strip() for c in args.test_conditions.split(',')] if args.test_conditions else None
    
    print("\n" + "="*80)
    print("RQ3: FINAL COMPLETE - REVISED VERSION")
    print("="*80)
    print("Incorporating reviewer feedback:")
    print("  A. Renamed 'Cohen's D' -> 'Relative Magnitude' (statistical correction)")
    print("  B. GPR logic: AND=min, OR=max (biological correction)")
    print(f"  C. Normalization strategy: {args.normalization_strategy}")
    print(f"  D. FVA for uncertainty: {'ENABLED' if args.run_fva else 'DISABLED'}")
    print("="*80 + "\n")
    
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Load data
    expression_matrix, metadata = load_single_cell_data(args.sc_data, args.sc_metadata)
    
    # Aggregate
    print("\n[STEP 1] Aggregating expression...")
    pseudo_bulk_dict, aggregation_summary = aggregate_expression(
        expression_matrix, metadata, min_cells=args.min_cells
    )
    aggregation_summary.to_csv(os.path.join(args.results_dir, "aggregation_summary.csv"), index=False)
    print(f"[INFO] Aggregation summary saved")
    
    # Load model
    print("\n[STEP 2] Loading model...")
    model = load_json_model(args.model)
    print(f"[INFO] Model: {model.id}, {len(model.reactions)} reactions, {len(model.genes)} genes")
    
    # Load dietary bounds
    print("\n[STEP 3] Loading dietary constraints...")
    diet_bounds_dict = load_dietary_bounds(args.diet_bounds_file, condition_mapping)
    
    # Apply CORRECT E-Flux
    print("\n[STEP 4] Applying CORRECT E-Flux + pFBA...")
    solutions_dict, fva_dict = apply_eflux_correctly(
        model, pseudo_bulk_dict, diet_bounds_dict,
        args.eflux_quantile, args.eflux_floor, args.eflux_cap,
        args.objective,
        args.normalization_strategy,
        args.run_fva,
        args.fva_fraction
    )
    
    # Compare fluxes
    print("\n[STEP 5] Comparing fluxes...")
    comparison_df = compare_fluxes(solutions_dict, model, args.results_dir)
    
    # Statistical testing (REVISED)
    print("\n[STEP 6] Statistical testing (REVISED)...")
    stats_df = perform_statistical_tests(
        solutions_dict, model, args.baseline, test_conditions,
        args.fold_change_threshold, args.abs_change_threshold, args.results_dir,
        fva_dict
    )
    
    # Visualizations
    print("\n[STEP 7] Creating visualizations...")
    create_visualizations(stats_df, solutions_dict, args.results_dir, fva_dict)
    
    # Final summary
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"Results directory: {args.results_dir}/")
    print(f"  - aggregation_summary.csv")
    print(f"  - statistics/flux_comparison.csv")
    print(f"  - statistics/statistical_tests.csv")
    print(f"  - statistics/METHODOLOGY_NOTES.txt READ THIS!")
    #print(f"  - statistics/METHODOLOGY_NOTES.txt  ← READ THIS!")
    print(f"  - visualizations/*.png")
    print("="*80)
    print("\nIMPORTANT REMINDERS:")
    print("  - 'relative_magnitude' is NOT Cohen's d (see methodology notes)")
    print("  - 'significant' means threshold-exceeded, not statistically significant")
    print("  - Solutions are deterministic (one per cell type × condition)")
    if args.run_fva:
        print("  - FVA ranges provide uncertainty bounds on flux values")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
