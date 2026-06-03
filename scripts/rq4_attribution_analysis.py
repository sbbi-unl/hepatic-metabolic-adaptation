#!/usr/bin/env python3
"""
RQ4: Differential Flux Attribution Analysis
============================================
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import cobra
from cobra.io import load_json_model


################################################################################
# ATTRIBUTION ANALYSIS
################################################################################

def load_flux_comparison_data(
    results_dir: str,
    conditions: List[str]
) -> Dict[str, pd.DataFrame]:
    """
    Load flux comparison data for multiple conditions.
    
    Returns dict: {condition: flux_comparison_df}
    """
    print("[INFO] Loading flux comparison data...")
    
    flux_data = {}
    
    for condition in conditions:
        # Check in condition subdirectory first (new structure)
        filepath = os.path.join(results_dir, f'condition_{condition}', f'{condition}_flux_comparison.csv') # f'condition_{condition}',
        
        # Fall back to old location if not found
        if not os.path.exists(filepath):
            filepath = os.path.join(results_dir,f'condition_{condition}',f'condition_{condition}', f'{condition}_flux_comparison.csv')
        
        if not os.path.exists(filepath):
            print(f"[WARNING] File not found: {filepath}")
            print(f"[WARNING] Also checked: {os.path.join(results_dir, f'condition_{condition}', f'condition_{condition}', f'{condition}_flux_comparison.csv')}")
            continue
        
        df = pd.read_csv(filepath)
        flux_data[condition] = df
        print(f"  [Done] Loaded {condition}: {len(df)} reactions")
        #print(f"  ✓ Loaded {condition}: {len(df)} reactions")
    
    return flux_data


def calculate_variance_components(
    flux_baseline_scd: pd.Series,
    flux_diet_hfd: pd.Series,
    flux_microbiome_hfd: pd.Series,
    reaction_ids: pd.Series
) -> pd.DataFrame:
    """
    Decompose flux variance into genetic, dietary, and microbiome components.

    """
    print("\n[INFO] Calculating comprehensive variance components...")
    
    attribution_data = []
    
    for i in range(len(reaction_ids)):
        rxn_id = reaction_ids.iloc[i]
        
        # Get flux values
        flux_scd = flux_baseline_scd.iloc[i]
        flux_hfd_no_mb = flux_diet_hfd.iloc[i]
        flux_hfd_with_mb = flux_microbiome_hfd.iloc[i]
        
        # ====================================================================
        # ABSOLUTE FLUX CHANGES (Option 1)
        # ====================================================================
        delta_diet = flux_hfd_no_mb - flux_scd  # Diet effect
        delta_microbiome = flux_hfd_with_mb - flux_hfd_no_mb  # Microbiome effect
        delta_total = flux_hfd_with_mb - flux_scd  # Total change
        
        abs_diet = abs(delta_diet)
        abs_microbiome = abs(delta_microbiome)
        abs_total = abs(delta_total)
        
        # ====================================================================
        # VARIANCE EXPLAINED - R² APPROACH (Option 3)
        # ====================================================================
        # Variance = squared deviations
        var_diet = delta_diet ** 2
        var_microbiome = delta_microbiome ** 2
        var_total = var_diet + var_microbiome
        
        # Percentage of variance explained
        if var_total > 1e-12:
            variance_explained_diet = (var_diet / var_total) * 100
            variance_explained_microbiome = (var_microbiome / var_total) * 100
        else:
            variance_explained_diet = 0.0
            variance_explained_microbiome = 0.0
        
        # ====================================================================
        # DIRECTIONAL ATTRIBUTION (Option 2)
        # ====================================================================
        # Check if diet and microbiome have same direction
        same_direction = (delta_diet * delta_microbiome) >= 0
        
        if abs_total < 1e-6:
            effect_type = 'Negligible'
            interaction = 'None'
        elif same_direction:
            effect_type = 'Synergistic'
            if abs(delta_diet) > abs(delta_microbiome):
                interaction = 'Diet-dominated synergy'
            elif abs(delta_microbiome) > abs(delta_diet):
                interaction = 'Microbiome-dominated synergy'
            else:
                interaction = 'Equal synergy'
        else:
            effect_type = 'Antagonistic'
            if abs_diet > 2 * abs_microbiome:
                interaction = 'Diet overcomes microbiome'
            elif abs_microbiome > 2 * abs_diet:
                interaction = 'Microbiome overcomes diet'
            else:
                interaction = 'Balanced opposition'
        
        # Flag opposing effects (causes >100% percentages)
        opposing_effects = not same_direction and abs_total > 1e-6
        
        # ====================================================================
        # EFFECT SIZES (Cohen's d)
        # ====================================================================
        # Effect size for diet: standardized change
        # Using total flux as scaling factor (rough approximation)
        if abs(flux_scd) > 1e-6:
            cohens_d_diet = delta_diet / abs(flux_scd)
        else:
            cohens_d_diet = 0.0
            
        if abs(flux_hfd_no_mb) > 1e-6:
            cohens_d_microbiome = delta_microbiome / abs(flux_hfd_no_mb)
        else:
            cohens_d_microbiome = 0.0
        
        # Classify effect size magnitude
        # |d| < 0.2: small, 0.2-0.8: medium, >0.8: large
        if abs(cohens_d_diet) < 0.2:
            diet_effect_size = 'Small'
        elif abs(cohens_d_diet) < 0.8:
            diet_effect_size = 'Medium'
        else:
            diet_effect_size = 'Large'
            
        if abs(cohens_d_microbiome) < 0.2:
            microbiome_effect_size = 'Small'
        elif abs(cohens_d_microbiome) < 0.8:
            microbiome_effect_size = 'Medium'
        else:
            microbiome_effect_size = 'Large'
        
        # ====================================================================
        # DOMINANT DRIVER (Improved classification)
        # ====================================================================
        if abs_total < 1e-3:
            dominant_driver = 'Stable'
        elif opposing_effects:
            if abs_diet > abs_microbiome:
                dominant_driver = 'Diet (opposed by microbiome)'
            else:
                dominant_driver = 'Microbiome (opposed by diet)'
        else:
            if abs_microbiome > abs_diet:
                dominant_driver = 'Microbiome'
            elif abs_diet > abs_microbiome:
                dominant_driver = 'Diet'
            else:
                dominant_driver = 'Equal'
        
        # ====================================================================
        # LEGACY METRICS (for backward compatibility)
        # ====================================================================
        # Old percentage calculation (can exceed 100% with opposing effects)
        if abs_total > 1e-6:
            pct_diet_old = (abs_diet / abs_total) * 100
            pct_microbiome_old = (abs_microbiome / abs_total) * 100
        else:
            pct_diet_old = 0.0
            pct_microbiome_old = 0.0
        
        # ====================================================================
        # STORE ALL METRICS
        # ====================================================================
        attribution_data.append({
            # Basic flux values
            'reaction_id': rxn_id,
            'flux_scd_baseline': flux_scd,
            'flux_hfd_no_microbiome': flux_hfd_no_mb,
            'flux_hfd_with_microbiome': flux_hfd_with_mb,
            
            # Absolute changes (mmol/gDW/h) - OPTION 1
            'delta_diet': delta_diet,
            'delta_microbiome': delta_microbiome,
            'delta_total': delta_total,
            'abs_diet_contribution': abs_diet,
            'abs_microbiome_contribution': abs_microbiome,
            
            # Variance explained (R² approach) - OPTION 3
            'variance_explained_diet': variance_explained_diet,
            'variance_explained_microbiome': variance_explained_microbiome,
            
            # Directional classification - OPTION 2
            'effect_type': effect_type,
            'interaction': interaction,
            'opposing_effects': opposing_effects,
            
            # Effect sizes (Cohen's d)
            'cohens_d_diet': cohens_d_diet,
            'cohens_d_microbiome': cohens_d_microbiome,
            'diet_effect_size': diet_effect_size,
            'microbiome_effect_size': microbiome_effect_size,
            
            # Dominant driver
            'dominant_driver': dominant_driver,
            
            # Legacy metrics (for comparison)
            'pct_diet_contribution_legacy': pct_diet_old,
            'pct_microbiome_contribution_legacy': pct_microbiome_old,
        })
    
    df = pd.DataFrame(attribution_data)
    
    # ====================================================================
    # COMPREHENSIVE SUMMARY STATISTICS
    # ====================================================================
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE ATTRIBUTION ANALYSIS")
    print(f"{'='*80}")
    
    print(f"\n[BASIC STATISTICS]")
    print(f"  Total reactions analyzed: {len(df)}")
    
    significant = df[df['delta_total'].abs() > 1e-3]
    print(f"  Reactions with significant changes: {len(significant)} ({len(significant)/len(df)*100:.1f}%)")
    
    # ====================================================================
    # OPTION 1: ABSOLUTE CONTRIBUTIONS
    # ====================================================================
    print(f"\n[ABSOLUTE FLUX CONTRIBUTIONS]")
    print(f"  Mean absolute contributions (mmol/gDW/h):")
    print(f"    Diet:       {significant['abs_diet_contribution'].mean():.3f} +/- {significant['abs_diet_contribution'].std():.3f}")
    print(f"    Microbiome: {significant['abs_microbiome_contribution'].mean():.3f} +/- {significant['abs_microbiome_contribution'].std():.3f}")
    print(f"  Median absolute contributions:")
    print(f"    Diet:       {significant['abs_diet_contribution'].median():.3f}")
    print(f"    Microbiome: {significant['abs_microbiome_contribution'].median():.3f}")
    
    # ====================================================================
    # OPTION 2: DIRECTIONAL EFFECTS
    # ====================================================================
    print(f"\n[DIRECTIONAL ATTRIBUTION]")
    synergistic = significant[significant['effect_type'] == 'Synergistic']
    antagonistic = significant[significant['effect_type'] == 'Antagonistic']
    
    print(f"  Synergistic effects (same direction): {len(synergistic)} ({len(synergistic)/len(significant)*100:.1f}%)")
    print(f"  Antagonistic effects (opposing):      {len(antagonistic)} ({len(antagonistic)/len(significant)*100:.1f}%)")
    
    if len(antagonistic) > 0:
        print(f"\n  [WARNING] {len(antagonistic)} reactions show opposing diet/microbiome effects!")
        print(f"            This causes legacy percentages to exceed 100%.")
        print(f"            Use variance_explained or absolute contributions for interpretation.")
    
    # ====================================================================
    # OPTION 3: VARIANCE EXPLAINED
    # ====================================================================
    print(f"\n[VARIANCE EXPLAINED (R² approach)]")
    print(f"  Mean variance explained (%):")
    print(f"    Diet:       {significant['variance_explained_diet'].mean():.1f}% +/- {significant['variance_explained_diet'].std():.1f}%")
    print(f"    Microbiome: {significant['variance_explained_microbiome'].mean():.1f}% +/- {significant['variance_explained_microbiome'].std():.1f}%")
    print(f"  Note: These percentages sum to 100% by construction")
    
    # ====================================================================
    # EFFECT SIZES
    # ====================================================================
    print(f"\n[EFFECT SIZE DISTRIBUTION]")
    diet_large = len(significant[significant['diet_effect_size'] == 'Large'])
    diet_medium = len(significant[significant['diet_effect_size'] == 'Medium'])
    diet_small = len(significant[significant['diet_effect_size'] == 'Small'])
    
    mb_large = len(significant[significant['microbiome_effect_size'] == 'Large'])
    mb_medium = len(significant[significant['microbiome_effect_size'] == 'Medium'])
    mb_small = len(significant[significant['microbiome_effect_size'] == 'Small'])
    
    print(f"  Diet effects:       Large={diet_large}, Medium={diet_medium}, Small={diet_small}")
    print(f"  Microbiome effects: Large={mb_large}, Medium={mb_medium}, Small={mb_small}")
    
    # ====================================================================
    # DOMINANT DRIVER
    # ====================================================================
    print(f"\n[DOMINANT DRIVER]")
    driver_counts = significant['dominant_driver'].value_counts()
    for driver, count in driver_counts.items():
        print(f"  {driver:40s}: {count:4d} ({count/len(significant)*100:5.1f}%)")
    
    return df


def pathway_level_attribution(
    attribution_df: pd.DataFrame,
    flux_comparison_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate attribution to pathway level.
    """
    print("\n[INFO] Calculating pathway-level attribution...")
    
    # Merge subsystem information
    if 'subsystem' in flux_comparison_df.columns:
        attribution_df = attribution_df.merge(
            flux_comparison_df[['reaction_id', 'subsystem']],
            on='reaction_id',
            how='left'
        )
    else:
        print("[WARNING] No subsystem information available")
        attribution_df['subsystem'] = 'Unknown'
    
    # Group by pathway
    pathway_stats = attribution_df.groupby('subsystem').agg({
        'delta_diet': ['mean', 'sum'],
        'delta_microbiome': ['mean', 'sum'],
        'delta_total': ['mean', 'sum'],
        'abs_diet_contribution': 'mean',
        'abs_microbiome_contribution': 'mean',
        'variance_explained_diet': 'mean',
        'variance_explained_microbiome': 'mean',
        'reaction_id': 'count'
    }).reset_index()
    
    pathway_stats.columns = ['_'.join(col).strip('_') for col in pathway_stats.columns.values]
    pathway_stats = pathway_stats.rename(columns={
        'subsystem': 'pathway',
        'reaction_id_count': 'n_reactions'
    })
    
    # Filter out pathways with only 1 reaction (likely noise)
    pathway_stats = pathway_stats[pathway_stats['n_reactions'] >= 2]
    
    # Sort by total flux change magnitude
    pathway_stats = pathway_stats.sort_values('delta_total_sum', key=abs, ascending=False)
    
    return pathway_stats


################################################################################
# STATISTICAL TESTING
################################################################################

def test_microbiome_significance(
    flux_hfd_no_mb: np.ndarray,
    flux_hfd_with_mb: np.ndarray,
    test_type: str = 'paired_t'
) -> Tuple[float, float]:
    """
    Test whether microbiome significantly affects hepatic fluxes.
    """
    # Remove reactions with zero flux in both conditions
    mask = ~((flux_hfd_no_mb == 0) & (flux_hfd_with_mb == 0))
    flux_no_mb_filtered = flux_hfd_no_mb[mask]
    flux_with_mb_filtered = flux_hfd_with_mb[mask]
    
    if test_type == 'paired_t':
        stat, pval = stats.ttest_rel(flux_with_mb_filtered, flux_no_mb_filtered)
    elif test_type == 'wilcoxon':
        #stat, pval = stats.wilcoxon(flux_with_mb_filtered, flux_no_mb_filtered)
        try:
            stat, pval = stats.wilcoxon(flux_with_mb_filtered, flux_no_mb_filtered)
        except ValueError as e:
            # If the arrays are perfectly identical, Wilcoxon throws a ValueError
            if "zero for all elements" in str(e).lower():
                stat, pval = 0.0, 1.0  # Safe default for identical data
            else:
                raise e
        
    else:
        raise ValueError(f"Unknown test_type: {test_type}")
    
    return stat, pval



def comprehensive_statistical_analysis(
    attribution_df: pd.DataFrame
) -> Dict:
    """
    Perform comprehensive statistical analysis on attribution data.
    
    """
    print(f"\n{'='*80}")
    print("COMPREHENSIVE STATISTICAL ANALYSIS")
    print(f"{'='*80}")
    
    results = {}
    
    # Filter to significant reactions
    significant = attribution_df[attribution_df["delta_total"].abs() > 1e-3]
    
    # CORRELATION ANALYSIS
    print(f"\n[CORRELATION ANALYSIS]")
    
    # Pearson correlation between diet and microbiome effects
    r_pearson, p_pearson = stats.pearsonr(
        significant["delta_diet"],
        significant["delta_microbiome"]
    )
    
    # Spearman correlation (non-parametric)
    r_spearman, p_spearman = stats.spearmanr(
        significant["delta_diet"],
        significant["delta_microbiome"]
    )
    
    print(f"  Pearson correlation:  r = {r_pearson:6.3f}, p = {p_pearson:.2e}")
    print(f"  Spearman correlation: rho = {r_spearman:6.3f}, p = {p_spearman:.2e}")
    
    if r_pearson > 0.3 and p_pearson < 0.05:
        print(f"  -> Diet and microbiome effects are POSITIVELY correlated (synergistic)")
    elif r_pearson < -0.3 and p_pearson < 0.05:
        print(f"  -> Diet and microbiome effects are NEGATIVELY correlated (antagonistic)")
    else:
        print(f"  -> Diet and microbiome effects are INDEPENDENT")
    
    results["correlation_pearson"] = r_pearson
    results["correlation_pearson_pval"] = p_pearson
    results["correlation_spearman"] = r_spearman
    results["correlation_spearman_pval"] = p_spearman
    
    # MAGNITUDE COMPARISON
    print(f"\n[MAGNITUDE COMPARISON]")
    
    abs_diet = significant["abs_diet_contribution"].values
    abs_mb = significant["abs_microbiome_contribution"].values
    
    t_stat, t_pval = stats.ttest_rel(abs_diet, abs_mb)
    w_stat, w_pval = stats.wilcoxon(abs_diet, abs_mb)
    
    print(f"  Testing: |Diet contribution| vs |Microbiome contribution|")
    print(f"    Paired t-test: t = {t_stat:.3f}, p = {t_pval:.3e}")
    print(f"    Wilcoxon test: W = {w_stat:.1f}, p = {w_pval:.3e}")
    
    if t_pval < 0.05:
        if abs_diet.mean() > abs_mb.mean():
            print(f"  -> Diet effects are SIGNIFICANTLY LARGER than microbiome effects")
        else:
            print(f"  -> Microbiome effects are SIGNIFICANTLY LARGER than diet effects")
    else:
        print(f"  -> No significant difference in magnitude")
    
    results["magnitude_t_stat"] = t_stat
    results["magnitude_t_pval"] = t_pval
    results["magnitude_w_stat"] = w_stat
    results["magnitude_w_pval"] = w_pval
    
    # EFFECT SIZE STATISTICS
    print(f"\n[EFFECT SIZE STATISTICS]")
    
    pooled_std = np.sqrt((abs_diet.var() + abs_mb.var()) / 2)
    if pooled_std > 0:
        cohens_d = (abs_diet.mean() - abs_mb.mean()) / pooled_std
    else:
        cohens_d = 0.0
    
    print(f"  Cohen's d (diet vs microbiome magnitude): {cohens_d:.3f}")
    if abs(cohens_d) < 0.2:
        print(f"  -> Small effect size difference")
    elif abs(cohens_d) < 0.8:
        print(f"  -> Medium effect size difference")
    else:
        print(f"  -> Large effect size difference")
    
    results["cohens_d_magnitude"] = cohens_d
    
    diet_d = significant["cohens_d_diet"].values
    mb_d = significant["cohens_d_microbiome"].values
    
    print(f"\n  Distribution of individual Cohen's d:")
    print(f"    Diet:       mean = {np.mean(diet_d):6.3f}, median = {np.median(diet_d):6.3f}")
    print(f"    Microbiome: mean = {np.mean(mb_d):6.3f}, median = {np.median(mb_d):6.3f}")
    
    # DIRECTIONAL CONSISTENCY
    print(f"\n[DIRECTIONAL CONSISTENCY]")
    
    n_synergistic = len(significant[significant["effect_type"] == "Synergistic"])
    n_antagonistic = len(significant[significant["effect_type"] == "Antagonistic"])
    
    synergistic_proportion = n_synergistic / len(significant)
    # Use binomtest for scipy >= 1.7 (binom_test deprecated)
    try:
        binom_result = stats.binomtest(n_synergistic, len(significant), p=0.5, alternative="greater")
        binom_test_pval = binom_result.pvalue
    except AttributeError:
        # Fallback for older scipy versions
        binom_test_pval = stats.binom_test(n_synergistic, len(significant), p=0.5, alternative="greater")
    
    print(f"  Synergistic: {n_synergistic}/{len(significant)} ({synergistic_proportion*100:.1f}%)")
    print(f"  Antagonistic: {n_antagonistic}/{len(significant)} ({(1-synergistic_proportion)*100:.1f}%)")
    print(f"  Binomial test (H0: 50/50 split): p = {binom_test_pval:.3e}")
    
    if binom_test_pval < 0.05:
        print(f"  -> Effects are PREDOMINANTLY SYNERGISTIC")
    else:
        print(f"  -> Effects show no directional bias")
    
    results["synergistic_proportion"] = synergistic_proportion
    results["directionality_pval"] = binom_test_pval
    
    # VARIANCE EXPLAINED ANALYSIS
    print(f"\n[VARIANCE EXPLAINED ANALYSIS]")
    
    var_diet = significant["variance_explained_diet"].values
    var_mb = significant["variance_explained_microbiome"].values
    
    print(f"  Mean variance explained:")
    print(f"    Diet:       {var_diet.mean():.1f}% (range: {var_diet.min():.1f}% - {var_diet.max():.1f}%)")
    print(f"    Microbiome: {var_mb.mean():.1f}% (range: {var_mb.min():.1f}% - {var_mb.max():.1f}%)")
    
    diet_dominant_var = np.sum(var_diet > var_mb)
    mb_dominant_var = np.sum(var_mb > var_diet)
    
    print(f"\n  Variance-based dominance:")
    print(f"    Diet-dominant:       {diet_dominant_var} reactions ({diet_dominant_var/len(significant)*100:.1f}%)")
    print(f"    Microbiome-dominant: {mb_dominant_var} reactions ({mb_dominant_var/len(significant)*100:.1f}%)")
    
    results["variance_diet_mean"] = var_diet.mean()
    results["variance_microbiome_mean"] = var_mb.mean()
    results["variance_diet_dominant"] = diet_dominant_var
    results["variance_mb_dominant"] = mb_dominant_var
    
    # BIOLOGICAL SIGNIFICANCE THRESHOLDS
    print(f"\n[BIOLOGICAL SIGNIFICANCE]")
    
    bio_sig_diet = np.sum(significant["abs_diet_contribution"] > 0.1)
    bio_sig_mb = np.sum(significant["abs_microbiome_contribution"] > 0.1)
    
    print(f"  Reactions with |delta flux| > 0.1 mmol/gDW/h:")
    print(f"    Diet:       {bio_sig_diet}/{len(significant)} ({bio_sig_diet/len(significant)*100:.1f}%)")
    print(f"    Microbiome: {bio_sig_mb}/{len(significant)} ({bio_sig_mb/len(significant)*100:.1f}%)")
    
    large_diet = np.sum(significant["diet_effect_size"] == "Large")
    large_mb = np.sum(significant["microbiome_effect_size"] == "Large")
    
    print(f"\n  Reactions with large effect size (|d| > 0.8):")
    print(f"    Diet:       {large_diet}/{len(significant)} ({large_diet/len(significant)*100:.1f}%)")
    print(f"    Microbiome: {large_mb}/{len(significant)} ({large_mb/len(significant)*100:.1f}%)")
    
    results["bio_sig_diet"] = bio_sig_diet
    results["bio_sig_microbiome"] = bio_sig_mb
    results["large_effect_diet"] = large_diet
    results["large_effect_microbiome"] = large_mb
    
    return results

################################################################################
# VISUALIZATION
################################################################################

def plot_attribution_summary(
    attribution_df: pd.DataFrame,
    pathway_stats: pd.DataFrame,
    output_dir: str
):
    """
    Generate comprehensive attribution visualizations.
    """
    print("\n[INFO] Generating attribution visualizations...")
    
    # Filter for significant changes
    significant = attribution_df[attribution_df['delta_total'].abs() > 0.01]
    
    # ========================================================================
    # Figure 1: Scatter plot - Diet vs. Microbiome contribution
    # ========================================================================
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Color by dominant driver
    colors_map = {
        'Microbiome': '#d62728',
        'Diet': '#1f77b4',
        'Equal': '#2ca02c',
        'None (stable)': '#7f7f7f'
    }
    
    colors = [colors_map.get(d, '#7f7f7f') for d in significant['dominant_driver']]
    
    scatter = ax.scatter(
        significant['variance_explained_diet'],
        significant['variance_explained_microbiome'],
        c=colors,
        alpha=0.6,
        s=50,
        edgecolors='black',
        linewidth=0.5
    )
    
    # Diagonal line (equal contribution)
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, linewidth=1, label='Equal contribution')
    
    ax.set_xlabel('Diet Contribution (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Microbiome Contribution (%)', fontsize=14, fontweight='bold')
    ax.set_title('Relative Contributions of Diet vs. Microbiome\nto Hepatic Flux Changes',
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-10, 110)
    ax.set_ylim(-10, 110)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d62728', alpha=0.6, edgecolor='black', label='Microbiome-dominant'),
        Patch(facecolor='#1f77b4', alpha=0.6, edgecolor='black', label='Diet-dominant'),
        Patch(facecolor='#2ca02c', alpha=0.6, edgecolor='black', label='Equal contribution')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'diet_vs_microbiome_contribution.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {fig_path}")
    
    # ========================================================================
    # Figure 2: Stacked bar chart - Pathway attribution
    # ========================================================================
    if not pathway_stats.empty:
        # Sort by absolute value of delta_total_sum
        pathway_stats['abs_delta'] = pathway_stats['delta_total_sum'].abs()
        top_pathways = pathway_stats.nlargest(15, 'abs_delta')
        top_pathways = top_pathways.drop('abs_delta', axis=1)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        pathway_names = top_pathways['pathway']
        diet_contrib = top_pathways['variance_explained_diet_mean']
        microbiome_contrib = top_pathways['variance_explained_microbiome_mean']
        
        x_pos = np.arange(len(pathway_names))
        
        ax.barh(x_pos, diet_contrib, label='Diet', color='#1f77b4', alpha=0.8)
        ax.barh(x_pos, microbiome_contrib, left=diet_contrib, label='Microbiome', 
                color='#d62728', alpha=0.8)
        
        ax.set_yticks(x_pos)
        ax.set_yticklabels(pathway_names, fontsize=11)
        ax.set_xlabel('Contribution to Pathway Flux Change (%)', fontsize=13, fontweight='bold')
        ax.set_title('Diet vs. Microbiome Contribution by Metabolic Pathway',
                     fontsize=15, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.set_xlim(0, 110)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        fig_path = os.path.join(output_dir, 'pathway_attribution_stacked.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[SAVED] {fig_path}")
    
    # ========================================================================
    # Figure 3: Distribution of microbiome contribution
    # ========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histogram
    ax1.hist(
        significant['variance_explained_microbiome'],
        bins=30,
        color='#d62728',
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5
    )
    ax1.axvline(
        significant['variance_explained_microbiome'].mean(),
        color='black',
        linestyle='--',
        linewidth=2,
        label=f'Mean = {significant["variance_explained_microbiome"].mean():.1f}%'
    )
    ax1.set_xlabel('Microbiome Variance Explained (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Reactions', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of Microbiome Variance Contribution', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # Cumulative distribution
    sorted_contrib = np.sort(significant['variance_explained_microbiome'])
    cumulative = np.arange(1, len(sorted_contrib) + 1) / len(sorted_contrib) * 100
    
    ax2.plot(sorted_contrib, cumulative, linewidth=2, color='#d62728')
    ax2.axhline(50, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axvline(
        np.median(significant['variance_explained_microbiome']),
        color='black',
        linestyle='--',
        linewidth=1,
        alpha=0.5,
        label=f'Median = {np.median(significant["variance_explained_microbiome"]):.1f}%'
    )
    ax2.set_xlabel('Microbiome Contribution (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative % of Reactions', fontsize=12, fontweight='bold')
    ax2.set_title('Cumulative Distribution', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'microbiome_contribution_distribution.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {fig_path}")


################################################################################
# MAIN WORKFLOW
################################################################################

def plot_improved_attribution_visualizations(
    attribution_df: pd.DataFrame,
    output_dir: str
):
    """
    Generate improved attribution visualizations using new metrics.
    """
    significant = attribution_df[attribution_df['delta_total'].abs() > 1e-3]
    
    # ====================================================================
    # FIGURE 1: Absolute Contributions Scatter Plot
    # ====================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Scatter plot with directional coloring
    synergistic = significant[significant['effect_type'] == 'Synergistic']
    antagonistic = significant[significant['effect_type'] == 'Antagonistic']
    
    ax1.scatter(
        synergistic['abs_diet_contribution'],
        synergistic['abs_microbiome_contribution'],
        alpha=0.6, s=50, c='#2ecc71', label='Synergistic', edgecolors='black', linewidth=0.5
    )
    ax1.scatter(
        antagonistic['abs_diet_contribution'],
        antagonistic['abs_microbiome_contribution'],
        alpha=0.6, s=50, c='#e74c3c', label='Antagonistic', edgecolors='black', linewidth=0.5
    )
    
    # Add diagonal line (equal contribution)
    max_val = max(significant['abs_diet_contribution'].max(), 
                   significant['abs_microbiome_contribution'].max())
    ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=2, label='Equal contribution')
    
    ax1.set_xlabel('|Diet Contribution| (mmol/gDW/h)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('|Microbiome Contribution| (mmol/gDW/h)', fontsize=12, fontweight='bold')
    ax1.set_title('Absolute Flux Contributions', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Right: Variance explained scatter
    ax2.scatter(
        synergistic['variance_explained_diet'],
        synergistic['variance_explained_microbiome'],
        alpha=0.6, s=50, c='#2ecc71', label='Synergistic', edgecolors='black', linewidth=0.5
    )
    ax2.scatter(
        antagonistic['variance_explained_diet'],
        antagonistic['variance_explained_microbiome'],
        alpha=0.6, s=50, c='#e74c3c', label='Antagonistic', edgecolors='black', linewidth=0.5
    )
    
    ax2.plot([0, 100], [100, 0], 'k--', alpha=0.3, linewidth=2, label='Sum = 100%')
    ax2.set_xlabel('Diet Variance Explained (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Microbiome Variance Explained (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Variance Partitioning (R² approach)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'absolute_contributions_and_variance.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {fig_path}")
    
    # ====================================================================
    # FIGURE 2: Effect Direction Summary
    # ====================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie chart: Effect types
    effect_counts = significant['effect_type'].value_counts()
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']
    ax1.pie(
        effect_counts.values,
        labels=effect_counts.index,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )
    ax1.set_title('Effect Direction Classification', fontsize=14, fontweight='bold')
    
    # Bar chart: Dominant driver
    driver_counts = significant['dominant_driver'].value_counts()
    ax2.bar(
        range(len(driver_counts)),
        driver_counts.values,
        color='#3498db',
        edgecolor='black',
        linewidth=1.5
    )
    ax2.set_xticks(range(len(driver_counts)))
    ax2.set_xticklabels(driver_counts.index, rotation=45, ha='right', fontsize=10)
    ax2.set_ylabel('Number of Reactions', fontsize=12, fontweight='bold')
    ax2.set_title('Dominant Driver Distribution', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'effect_direction_summary.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {fig_path}")
    
    # ====================================================================
    # FIGURE 3: Effect Size Distributions
    # ====================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histogram: Cohen's d distribution
    ax1.hist(
        significant['cohens_d_diet'],
        bins=30,
        alpha=0.7,
        color='#3498db',
        edgecolor='black',
        linewidth=1,
        label='Diet'
    )
    ax1.hist(
        significant['cohens_d_microbiome'],
        bins=30,
        alpha=0.7,
        color='#e67e22',
        edgecolor='black',
        linewidth=1,
        label='Microbiome'
    )
    ax1.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax1.set_xlabel("Cohen's d", fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Reactions', fontsize=12, fontweight='bold')
    ax1.set_title("Effect Size Distribution", fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    
    # Stacked bar: Effect size categories
    effect_size_data = {
        'Diet': [
            len(significant[significant['diet_effect_size'] == 'Small']),
            len(significant[significant['diet_effect_size'] == 'Medium']),
            len(significant[significant['diet_effect_size'] == 'Large'])
        ],
        'Microbiome': [
            len(significant[significant['microbiome_effect_size'] == 'Small']),
            len(significant[significant['microbiome_effect_size'] == 'Medium']),
            len(significant[significant['microbiome_effect_size'] == 'Large'])
        ]
    }
    
    x = np.arange(2)
    width = 0.6
    colors_sizes = ['#95a5a6', '#f39c12', '#e74c3c']
    
    bottom = np.zeros(2)
    for i, size in enumerate(['Small', 'Medium', 'Large']):
        values = [effect_size_data['Diet'][i], effect_size_data['Microbiome'][i]]
        ax2.bar(x, values, width, label=size, bottom=bottom, color=colors_sizes[i], edgecolor='black', linewidth=1)
        bottom += values
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Diet', 'Microbiome'], fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Reactions', fontsize=12, fontweight='bold')
    ax2.set_title('Effect Size Categories', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'effect_size_distributions.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {fig_path}")
    
    # ====================================================================
    # FIGURE 4: Correlation Analysis
    # ====================================================================
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Hexbin plot for density visualization
    hexbin = ax.hexbin(
        significant['delta_diet'],
        significant['delta_microbiome'],
        gridsize=30,
        cmap='YlOrRd',
        mincnt=1,
        edgecolors='black',
        linewidths=0.2
    )
    
    # Add regression line
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(
        significant['delta_diet'],
        significant['delta_microbiome']
    )
    x_line = np.array([significant['delta_diet'].min(), significant['delta_diet'].max()])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'b--', linewidth=2, label=f'r = {r_value:.3f}, p = {p_value:.2e}')
    
    # Zero lines
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Diet Effect (Δflux, mmol/gDW/h)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Microbiome Effect (Δflux, mmol/gDW/h)', fontsize=12, fontweight='bold')
    ax.set_title('Correlation: Diet vs Microbiome Effects', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(hexbin, ax=ax)
    cbar.set_label('Reaction Density', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'diet_microbiome_correlation.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {fig_path}")



def main():
    parser = argparse.ArgumentParser(
        description="RQ4 Differential Flux Attribution Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Scientific Workflow:
-------------------
This script performs three-way attribution analysis to partition
hepatic flux variance into:
1. Genetic effects (baseline metabolic capacity)
2. Dietary effects (nutrient availability)  
3. Microbiome effects (portal metabolite delivery)

Required inputs:
- Flux data from baseline SCD condition (genetics)
- Flux data from HFD without microbiome (genetics + diet)
- Flux data from HFD with microbiome (genetics + diet + microbiome)

Outputs:
- Reaction-level attribution table
- Pathway-level attribution summary
- Statistical significance tests
- Publication-quality figures

Examples:
--------
python rq4_attribution_analysis.py \\
    --hepatic_results_dir results_rq4_hepatic_integration \\
    --baseline_condition SCD \\
    --treatment_condition HFD \\
    --results_dir results_rq4_attribution
"""
    )
    
    # Input
    parser.add_argument(
        '--hepatic_results_dir',
        required=True,
        help='Directory containing hepatic integration results'
    )
    parser.add_argument(
        '--baseline_condition',
        default='SCD',
        help='Baseline condition name (e.g., SCD, ND_SCD)'
    )
    parser.add_argument(
        '--treatment_condition',
        default='HFD',
        help='Treatment condition name (e.g., HFD, DD_HFD)'
    )
    
    # Output
    parser.add_argument(
        '--results_dir',
        default='results_rq4_attribution',
        help='Output directory for attribution analysis'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    print("="*80)
    print("RQ4: DIFFERENTIAL FLUX ATTRIBUTION ANALYSIS")
    print("="*80)
    
    # Load flux comparison data
    conditions = [args.baseline_condition, args.treatment_condition]
    flux_data = load_flux_comparison_data(args.hepatic_results_dir, conditions)
    
    if len(flux_data) < 2:
        print("[ERROR] Need both baseline and treatment condition data!")
        sys.exit(1)
    
    baseline_df = flux_data[args.baseline_condition]
    treatment_df = flux_data[args.treatment_condition]
    
    # Extract flux values
    # Assuming columns: flux_baseline (no microbiome), flux_microbiome (with microbiome)
    
    print(f"\n[INFO] Baseline condition: {args.baseline_condition}")
    print(f"[INFO] Treatment condition: {args.treatment_condition}")
    
    # Merge dataframes on reaction_id
    merged_df = baseline_df[['reaction_id', 'flux_baseline']].merge(
        treatment_df[['reaction_id', 'flux_baseline', 'flux_microbiome']],
        on='reaction_id',
        suffixes=('_scd', '_hfd')
    )
    
    # Calculate attribution
    attribution_df = calculate_variance_components(
        flux_baseline_scd=merged_df['flux_baseline_scd'],
        flux_diet_hfd=merged_df['flux_baseline_hfd'],
        flux_microbiome_hfd=merged_df['flux_microbiome'],
        reaction_ids=merged_df['reaction_id']
    )
    
    # Save attribution results
    attribution_path = os.path.join(args.results_dir, 'flux_attribution_analysis.csv')
    attribution_df.to_csv(attribution_path, index=False)
    print(f"\n[SAVED] Attribution analysis: {attribution_path}")
    
    # Pathway-level attribution
    pathway_stats = pathway_level_attribution(attribution_df, treatment_df)
    
    if not pathway_stats.empty:
        pathway_path = os.path.join(args.results_dir, 'pathway_attribution_summary.csv')
        pathway_stats.to_csv(pathway_path, index=False)
        print(f"[SAVED] Pathway attribution: {pathway_path}")
    
    # Statistical testing
    print("\n[INFO] Statistical significance testing...")
    stat_t, pval_t = test_microbiome_significance(
        merged_df['flux_baseline_hfd'].values,
        merged_df['flux_microbiome'],
        test_type='paired_t'
    )
    
    stat_w, pval_w = test_microbiome_significance(
        merged_df['flux_baseline_hfd'].values,
        merged_df['flux_microbiome'],
        test_type='wilcoxon'
    )
    
    print(f"  Paired t-test: t={stat_t:.4f}, p={pval_t:.4e}")
    print(f"  Wilcoxon test: W={stat_w:.4f}, p={pval_w:.4e}")
    
    if pval_t < 0.05:
        print("  -> Microbiome significantly affects hepatic fluxes (p < 0.05)")
    else:
        print("  -> No significant microbiome effect detected")
    
    # Generate visualizations
    plot_attribution_summary(attribution_df, pathway_stats, args.results_dir)
    
    # NEW: Comprehensive statistical analysis
    print("\n" + "="*80)
    stats_results = comprehensive_statistical_analysis(attribution_df)
    
    # NEW: Generate improved visualizations
    print("\n[INFO] Generating improved attribution visualizations...")
    plot_improved_attribution_visualizations(attribution_df, args.results_dir)
    
    # Save statistical results
    stats_path = os.path.join(args.results_dir, 'comprehensive_statistics.json')
    with open(stats_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        stats_json = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                      for k, v in stats_results.items()}
        json.dump(stats_json, f, indent=2)
    print(f"[SAVED] Statistical results: {stats_path}")
    
    # Final summary report (UPDATED WITH NEW METRICS)
    print("\n" + "="*80)
    print("PUBLICATION-READY ATTRIBUTION SUMMARY")
    print("="*80)
    
    significant = attribution_df[attribution_df['delta_total'].abs() > 1e-3]
    
    print(f"\nReactions with significant flux changes: {len(significant)}/{len(attribution_df)}")
    
    # === ABSOLUTE CONTRIBUTIONS ===
    print(f"\n[ABSOLUTE FLUX CONTRIBUTIONS]")
    print(f"  Mean +/- SD (mmol/gDW/h):")
    print(f"    Diet:       {significant['abs_diet_contribution'].mean():.3f} +/- {significant['abs_diet_contribution'].std():.3f}")
    print(f"    Microbiome: {significant['abs_microbiome_contribution'].mean():.3f} +/- {significant['abs_microbiome_contribution'].std():.3f}")
    
    # === VARIANCE EXPLAINED ===
    print(f"\n[VARIANCE EXPLAINED] (R² approach - sums to 100%)")
    print(f"  Mean (%):")
    print(f"    Diet:       {significant['variance_explained_diet'].mean():.1f}%")
    print(f"    Microbiome: {significant['variance_explained_microbiome'].mean():.1f}%")
    
    # === DIRECTIONAL EFFECTS ===
    print(f"\n[DIRECTIONAL EFFECTS]")
    n_synergistic = len(significant[significant['effect_type'] == 'Synergistic'])
    n_antagonistic = len(significant[significant['effect_type'] == 'Antagonistic'])
    print(f"  Synergistic (same direction):  {n_synergistic} ({n_synergistic/len(significant)*100:.1f}%)")
    print(f"  Antagonistic (opposing):       {n_antagonistic} ({n_antagonistic/len(significant)*100:.1f}%)")
    
    # === DOMINANT DRIVER ===
    print(f"\n[DOMINANT DRIVER]")
    driver_counts = significant['dominant_driver'].value_counts()
    for driver, count in driver_counts.items():
        print(f"  {driver}: {count} reactions ({count/len(significant)*100:.1f}%)")
    
    # === TOP REACTIONS ===
    print(f"\n[TOP 10 REACTIONS BY ABSOLUTE MICROBIOME CONTRIBUTION]")
    top_mb = significant.nlargest(10, 'abs_microbiome_contribution')
    for idx, row in top_mb.iterrows():
        print(f"  {row['reaction_id']:40s} delta = {row['delta_microbiome']:+7.3f} mmol/gDW/h ({row['microbiome_effect_size']} effect)")
    
    print("\n" + "="*80)
    print("RQ4 ATTRIBUTION ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {args.results_dir}")
    print(f"\nKey findings:")
    print(f"  - {len(significant)} reactions showed significant changes")
    print(f"  - Diet effects: {significant['variance_explained_diet'].mean():.1f}% of variance")
    print(f"  - Microbiome effects: {significant['variance_explained_microbiome'].mean():.1f}% of variance")
    print(f"  - {n_synergistic} reactions show synergistic effects")
    print(f"  - {n_antagonistic} reactions show antagonistic effects (explains >100% legacy percentages)")


if __name__ == "__main__":
    main()


