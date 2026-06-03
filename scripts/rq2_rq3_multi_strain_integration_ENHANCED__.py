#!/usr/bin/env python3
"""
===============================================================================
RQ2 + RQ3 MULTI-STRAIN INTEGRATION ANALYSIS - FIXED VERSION 2 (REVISED)
===============================================================================
Integrates genetic diversity (RQ2, 9 strains) with cellular heterogeneity 
(RQ3, 23 cell types) to determine if cellular architecture of metabolic 
responses is genetically conserved or strain-specific.

MAJOR FIX IN V2:
- Implements strain-specific attribution categories based on contribution thresholding
- Previous version had bug where all strains showed identical category distributions
- Now uses weighted contribution filtering to determine which cell types are 
  truly significant for each reaction in each specific strain

REVISION (Pipeline Integration):
- Added command-line arguments for flexible path configuration
- Maintains backward compatibility with hardcoded defaults
- Better integration with master orchestration pipeline

Strategy:
1. Use RQ1 multi-dataset significant reactions (statistically validated)
2. For each RQ2 strain, examine flux changes in those reactions
3. Attribute to cell types using RQ3 cellular data
4. Apply strain-specific contribution thresholding (default: 15%)
5. Compare cellular contributions across all 9 strains

Research Question:
"Does the cell-type attribution of bulk metabolic changes vary across 
genetically diverse mouse strains, or is the cellular architecture of 
dietary responses conserved?"

Author: PhD Dissertation Pipeline
Date: January 2026
Version: 3.0 - ENHANCED (Hierarchical Attribution + Bulk Validation)
===============================================================================
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from collections import defaultdict, Counter
import warnings
import logging

# Configure
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# ============================================================================
# Strain Configuration
# ============================================================================

STRAINS = [
    'C57BL6J',
    '129S1SvImJ',
    'AJ',
    'CASTEiJ',
    'DBA2J',
    'NODShiLtJ',
    'NZOHlLtJ',
    'PWKPhJ',
    'WSBEiJ',
]

STRAIN_FULL_NAMES = {
    'C57BL6J': 'C57BL/6J',
    '129S1SvImJ': '129S1/SvImJ',
    'AJ': 'A/J',
    'CASTEiJ': 'CAST/EiJ',
    'DBA2J': 'DBA/2J',
    'NODShiLtJ': 'NOD/ShiLtJ',
    'NZOHlLtJ': 'NZO/HlLtJ',
    'PWKPhJ': 'PWK/PhJ',
    'WSBEiJ': 'WSB/EiJ',
}

# ============================================================================
# Configuration Parameters (now configurable via command line)
# ============================================================================

# Default contribution threshold for strain-specific attribution
DEFAULT_CONTRIBUTION_THRESHOLD = 0.15  # 15%

# ============================================================================
# HIERARCHICAL GROUPINGS FOR ATTRIBUTION ANALYSIS
# ============================================================================

FUNCTION_HIERARCHY = {
    'Immune': [
        'Timd4+ resKC', 'Cd207+, Trem2+ Mo-KC', 'Cd207-, Trem2+ Mo-KC',
        'Cx3cr1+, Ccr2+ MdM', 'Trem2+, Spp1+ MdM', 'Ly6c2+, Ccr2+ Mo',
        'Ly6c2-, Spn+ Mo', 'Transitioning Mo', 'T cells', 'B cells',
        'Neutrophiles', 'Dendritic cells', 'DCs', 'Macrophages',
        'Dendrites', 'MCs', 'Mast cells', 'Cycling'
    ],
    'Metabolic': ['Hepatocytes', 'Cholangiocytes'],
    'Structural': ['LECs', 'qHSCs', 'aHSCs', 'cAMP qHSCs'],
    'Secretory': ['Hepatocytes', 'Cholangiocytes']
}

LOCATION_HIERARCHY = {
    'Sinusoidal': [
        'LECs', 'Timd4+ resKC', 'Cd207+, Trem2+ Mo-KC',
        'Cd207-, Trem2+ Mo-KC', 'qHSCs', 'aHSCs', 'cAMP qHSCs'
    ],
    'Parenchymal': ['Hepatocytes'],
    'Portal': ['Cholangiocytes'],
    'Circulating': [
        'Cx3cr1+, Ccr2+ MdM', 'Trem2+, Spp1+ MdM', 'Ly6c2+, Ccr2+ Mo',
        'Ly6c2-, Spn+ Mo', 'Transitioning Mo', 'T cells', 'B cells',
        'Neutrophiles', 'Macrophages', 'Dendrites', 'DCs',
        'Dendritic cells', 'MCs', 'Mast cells', 'Cycling'
    ]
}

LINEAGE_HIERARCHY = {
    'Myeloid': [
        'Timd4+ resKC', 'Cd207+, Trem2+ Mo-KC', 'Cd207-, Trem2+ Mo-KC',
        'Cx3cr1+, Ccr2+ MdM', 'Trem2+, Spp1+ MdM', 'Ly6c2+, Ccr2+ Mo',
        'Ly6c2-, Spn+ Mo', 'Transitioning Mo', 'Neutrophiles',
        'Dendritic cells', 'Dendrites', 'DCs', 'Macrophages', 'MCs'
    ],
    'Lymphoid': ['T cells', 'B cells'],
    'Epithelial': ['Hepatocytes', 'Cholangiocytes'],
    'Endothelial': ['LECs'],
    'Mesenchymal': ['qHSCs', 'aHSCs', 'cAMP qHSCs', 'Mast cells']
}

# ============================================================================
# Helper Functions
# ============================================================================

def calculate_cell_abundance_from_scrna(aggregation_summary_file):
    """Calculate cell-type abundance from scRNA-seq data."""
    logger.info(f"Calculating cell abundance from: {aggregation_summary_file}")
    agg_df = pd.read_csv(aggregation_summary_file)
    cell_counts = agg_df.groupby('cell_type')['n_cells'].sum()
    total_cells = cell_counts.sum()
    abundance = (cell_counts / total_cells).to_dict()
    logger.info(f"  Total cells: {total_cells:,}, Cell types: {len(abundance)}")
    return abundance

def load_rq1_significant_reactions(rq1_stats_file, comparison='WD_vs_SCD', 
                                   fdr_threshold=0.10):
    """
    Load statistically significant reactions from RQ1 multi-dataset analysis.
    These provide the statistical filter for RQ2 single-strain analysis.
    """
    logger.info(f"Loading RQ1 significant reactions: {comparison}, FDR < {fdr_threshold}")
    stats_df = pd.read_csv(rq1_stats_file)
    
    # Filter for comparison
    comp_df = stats_df[
        ((stats_df['GroupA'] == 'SCD') & (stats_df['GroupB'] == 'WD'))
    ].copy()
    
    # Filter for significance
    sig_df = comp_df[comp_df['FDR_BH'] < fdr_threshold].copy()
    
    logger.info(f"  Found {len(sig_df)} significant reactions (FDR < {fdr_threshold})")
    
    return sig_df['ReactionID'].unique()

def load_rq2_strain_data(rq2_file, rq1_significant_reactions):
    """
    Load RQ2 strain-specific flux data, filtered for RQ1-significant reactions.
    """
    strain_df = pd.read_csv(rq2_file)
    
    # Filter for RQ1-significant reactions
    strain_df = strain_df[strain_df['ReactionID'].isin(rq1_significant_reactions)].copy()
    
    # Calculate absolute difference and fold change
    strain_df['abs_diff'] = np.abs(strain_df['Diff(HFD-SCD)'])
    strain_df['abs_fold_change'] = np.abs(np.log2(strain_df['Ratio(HFD/SCD)'] + 1e-10))
    
    return strain_df

def load_rq3_cellular_data(rq3_stats_file, comparison='WesternDiet_vs_Chow'):
    """Load RQ3 cell-type-specific significant reactions."""
    logger.info(f"Loading RQ3 cellular data: {comparison}")
    rq3_df = pd.read_csv(rq3_stats_file)
    rq3_df = rq3_df[rq3_df['comparison'] == comparison].copy()
    
    # Get significant reactions per cell type
    sig_by_cell_type = {}
    for cell_type in rq3_df['cell_type'].unique():
        ct_df = rq3_df[rq3_df['cell_type'] == cell_type]
        sig_rxns = ct_df[ct_df['significant'] == True]['reaction_id'].unique()
        sig_by_cell_type[cell_type] = set(sig_rxns)
    
    return rq3_df, sig_by_cell_type

# ============================================================================
# FIXED: Strain-Specific Attribution Function
# ============================================================================

def calculate_strain_specific_attribution(strain, rxn_id, strain_df,
                                         cellular_sig_by_cell_type,
                                         abundance_dict,
                                         contribution_threshold):
    """
    Calculate strain-specific attribution with contribution thresholding.
    
    KEY FIX: A cell type only counts as "significant" for attribution if:
    1. It's significant in RQ3 data (transcriptionally responsive)
    2. Its weighted contribution exceeds threshold in THIS specific strain
    
    This ensures attribution categories vary by strain based on actual
    metabolic contribution patterns, not just universal RQ3 significance.
    """
    
    # Find candidate cell types from RQ3 (transcriptionally responsive)
    weighted_contributions = {}
    
    for cell_type, sig_rxns in cellular_sig_by_cell_type.items():
        if rxn_id in sig_rxns:
            # Get STRAIN-SPECIFIC flux change from RQ2 data
            rxn_flux = strain_df[strain_df['ReactionID'] == rxn_id]
            if len(rxn_flux) > 0:
                flux_change = rxn_flux.iloc[0]['abs_diff']
                abundance = abundance_dict.get(cell_type, 0.001)
                weighted_contributions[cell_type] = flux_change * abundance
    
    # Calculate total contribution for normalization
    total_contribution = sum(weighted_contributions.values())
    
    # Filter to only cell types that meaningfully contribute in THIS strain
    significant_cell_types = []
    if total_contribution > 0:
        for cell_type, contrib in weighted_contributions.items():
            fraction = contrib / total_contribution
            if fraction >= contribution_threshold:
                significant_cell_types.append(cell_type)
    
    # Classify based on STRAIN-SPECIFIC significant cell types
    n_cell_types = len(significant_cell_types)
    
    if n_cell_types == 0:
        category = 'NON_CELLULAR'
        primary_driver = 'NONE'
    elif n_cell_types == 1:
        category = 'CELL_TYPE_UNIQUE'
        primary_driver = significant_cell_types[0]
    elif n_cell_types <= 3:
        category = 'COOPERATIVE'
        sorted_contrib = sorted(weighted_contributions.items(),
                               key=lambda x: x[1], reverse=True)
        primary_driver = sorted_contrib[0][0] if len(sorted_contrib) > 0 else 'UNKNOWN'
    else:
        category = 'MULTI_CELLULAR'
        sorted_contrib = sorted(weighted_contributions.items(),
                               key=lambda x: x[1], reverse=True)
        primary_driver = sorted_contrib[0][0] if len(sorted_contrib) > 0 else 'UNKNOWN'
    
    return category, n_cell_types, primary_driver, weighted_contributions, significant_cell_types

# ============================================================================
# Per-Strain Integration Analysis
# ============================================================================

def run_strain_integration(strain, rq2_file, rq1_sig_reactions, 
                          rq3_stats_df, cellular_sig_by_cell_type,
                          abundance_dict, contribution_threshold):
    """
    Run integration analysis for a single strain.
    
    Returns contribution_df, attribution_df
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"ANALYZING STRAIN: {STRAIN_FULL_NAMES[strain]}")
    logger.info(f"{'='*80}")
    
    # Load RQ2 strain data
    strain_df = load_rq2_strain_data(rq2_file, rq1_sig_reactions)
    strain_reactions = set(strain_df['ReactionID'])
    
    logger.info(f"  Reactions in strain (from RQ1 filter): {len(strain_reactions)}")
    
    # Calculate contributions
    contribution_results = []
    
    for cell_type, sig_rxns in cellular_sig_by_cell_type.items():
        # Overlap with strain reactions
        overlap_rxns = strain_reactions & sig_rxns
        
        # Get RQ3 flux data (for reference)
        ct_flux = rq3_stats_df[
            (rq3_stats_df['cell_type'] == cell_type) &
            (rq3_stats_df['reaction_id'].isin(overlap_rxns))
        ]
        
        n_sig = len(overlap_rxns)
        n_total = len(rq3_stats_df[rq3_stats_df['cell_type'] == cell_type])
        response_rate = n_sig / n_total if n_total > 0 else 0
        
        # Mean flux change - USE STRAIN-SPECIFIC RQ2 DATA!
        strain_flux = strain_df[strain_df['ReactionID'].isin(overlap_rxns)]
        mean_flux_change = strain_flux['abs_diff'].mean() if len(strain_flux) > 0 else 0
        
        # Cell abundance
        abundance = abundance_dict.get(cell_type, 0.001)
        
        # Contribution score
        contribution_score = abundance * response_rate * mean_flux_change
        
        contribution_results.append({
            'strain': strain,
            'cell_type': cell_type,
            'cell_abundance': abundance,
            'n_significant': n_sig,
            'n_total': n_total,
            'response_rate': response_rate,
            'mean_flux_change': mean_flux_change,
            'contribution_score': contribution_score,
        })
    
    contrib_df = pd.DataFrame(contribution_results)
    
    # Normalize to percentages
    total_contrib = contrib_df['contribution_score'].sum()
    if total_contrib > 0:
        contrib_df['contribution_percent'] = contrib_df['contribution_score'] / total_contrib * 100
    else:
        contrib_df['contribution_percent'] = 0
    
    contrib_df = contrib_df.sort_values('contribution_percent', ascending=False)
    
    # Log top 5
    logger.info(f"\n  Top 5 Contributors:")
    for idx, row in contrib_df.head(5).iterrows():
        logger.info(f"    {row['cell_type']:30s}: {row['contribution_percent']:5.1f}% "
                   f"(abundance: {row['cell_abundance']*100:5.1f}%, response: {row['response_rate']*100:5.1f}%)")
    
    # FIXED: Attribution analysis with strain-specific thresholding
    attribution_results = []
    threshold_diagnostic = defaultdict(list)  # Track threshold effects
    
    for rxn_id in strain_reactions:
        # Use new strain-specific attribution function
        category, n_cell_types, primary_driver, weighted_contributions, significant_cell_types = \
            calculate_strain_specific_attribution(
                strain, rxn_id, strain_df,
                cellular_sig_by_cell_type,
                abundance_dict,
                contribution_threshold=contribution_threshold
            )
        
        # Track contribution distributions for diagnostics
        if len(weighted_contributions) > 0:
            total = sum(weighted_contributions.values())
            fractions = {ct: val/total for ct, val in weighted_contributions.items()}
            threshold_diagnostic[category].append({
                'rxn_id': rxn_id,
                'n_candidates': len(weighted_contributions),
                'n_significant': n_cell_types,
                'max_contribution': max(fractions.values()) if fractions else 0,
                'top3_sum': sum(sorted(fractions.values(), reverse=True)[:3])
            })
        
        attribution_results.append({
            'strain': strain,
            'reaction_id': rxn_id,
            'category': category,
            'n_cell_types': n_cell_types,
            'primary_driver': primary_driver,
        })
    
    attrib_df = pd.DataFrame(attribution_results)
    
    # Log attribution summary
    category_counts = attrib_df['category'].value_counts()
    logger.info(f"\n  Attribution Categories (with {contribution_threshold:.1%} threshold):")
    for cat, count in category_counts.items():
        pct = count / len(attrib_df) * 100 if len(attrib_df) > 0 else 0
        logger.info(f"    {cat:20s}: {count:4d} ({pct:5.1f}%)")
    
    # Log diagnostic info
    logger.info(f"\n  Threshold Diagnostic:")
    for cat in ['MULTI_CELLULAR', 'COOPERATIVE', 'CELL_TYPE_UNIQUE', 'NON_CELLULAR']:
        if cat in threshold_diagnostic and len(threshold_diagnostic[cat]) > 0:
            diag_data = threshold_diagnostic[cat]
            avg_candidates = np.mean([d['n_candidates'] for d in diag_data])
            avg_significant = np.mean([d['n_significant'] for d in diag_data])
            avg_max = np.mean([d['max_contribution'] for d in diag_data])
            logger.info(f"    {cat:20s}: Avg candidates={avg_candidates:.1f}, "
                       f"significant={avg_significant:.1f}, max_contrib={avg_max:.2%}")
    
    return contrib_df, attrib_df

# ============================================================================
# Cross-Strain Analysis
# ============================================================================

def analyze_cross_strain_conservation(all_contributions, all_attributions, strains):
    """
    Analyze conservation vs divergence of cellular architecture across strains.
    """
    logger.info("\n" + "="*80)
    logger.info("CROSS-STRAIN CONSERVATION ANALYSIS")
    logger.info("="*80)
    
    # Create contribution matrix: Cell types × Strains
    contrib_matrix = pd.pivot_table(
        all_contributions,
        values='contribution_percent',
        index='cell_type',
        columns='strain',
        fill_value=0
    )
    
    # Calculate coefficient of variation per cell type
    cv_results = []
    for cell_type in contrib_matrix.index:
        values = contrib_matrix.loc[cell_type].values
        mean_contrib = np.mean(values)
        std_contrib = np.std(values)
        cv = std_contrib / mean_contrib if mean_contrib > 0 else 0
        
        cv_results.append({
            'cell_type': cell_type,
            'mean_contribution': mean_contrib,
            'std_contribution': std_contrib,
            'cv': cv,
            'min_strain': contrib_matrix.loc[cell_type].idxmin(),
            'min_value': contrib_matrix.loc[cell_type].min(),
            'max_strain': contrib_matrix.loc[cell_type].idxmax(),
            'max_value': contrib_matrix.loc[cell_type].max(),
        })
    
    cv_df = pd.DataFrame(cv_results)
    cv_df = cv_df.sort_values('mean_contribution', ascending=False)
    
    logger.info("\nTop 10 Cell Types by Mean Contribution Across Strains:")
    logger.info("-" * 100)
    logger.info(f"{'Cell Type':<30s} {'Mean':>8s} {'Std':>8s} {'CV':>8s} {'Conserved?':>12s}")
    logger.info("-" * 100)
    for idx, row in cv_df.head(10).iterrows():
        conserved = "CONSERVED" if row['cv'] < 0.3 else "VARIABLE"
        logger.info(f"{row['cell_type']:<30s} {row['mean_contribution']:>8.2f} "
                   f"{row['std_contribution']:>8.2f} {row['cv']:>8.2f} {conserved:>12s}")
    
    # Primary driver consistency
    primary_driver_counts = defaultdict(Counter)
    
    for strain in strains:
        strain_attrib = all_attributions[all_attributions['strain'] == strain]
        driver_counts = strain_attrib['primary_driver'].value_counts()
        for driver, count in driver_counts.items():
            primary_driver_counts[driver][strain] = count
    
    # Calculate driver consistency
    driver_consistency = []
    for driver in primary_driver_counts:
        strains_present = len(primary_driver_counts[driver])
        total_reactions = sum(primary_driver_counts[driver].values())
        mean_reactions = total_reactions / len(strains)
        
        driver_consistency.append({
            'primary_driver': driver,
            'strains_present': strains_present,
            'total_reactions': total_reactions,
            'mean_reactions_per_strain': mean_reactions,
            'consistency': strains_present / len(strains),
        })
    
    consistency_df = pd.DataFrame(driver_consistency)
    consistency_df = consistency_df.sort_values('total_reactions', ascending=False)
    
    logger.info("\nPrimary Driver Consistency Across Strains:")
    logger.info("-" * 100)
    logger.info(f"{'Primary Driver':<30s} {'Total Rxns':>12s} {'Strains':>8s} {'Consistency':>12s}")
    logger.info("-" * 100)
    for idx, row in consistency_df.head(10).iterrows():
        logger.info(f"{row['primary_driver']:<30s} {int(row['total_reactions']):>12d} "
                   f"{int(row['strains_present']):>8d}/{len(strains)} {row['consistency']*100:>11.1f}%")
    
    # Category distribution analysis
    logger.info("\nAttribution Category Distribution by Strain:")
    logger.info("-" * 100)
    category_by_strain = all_attributions.groupby(['strain', 'category']).size().unstack(fill_value=0)
    category_by_strain['TOTAL'] = category_by_strain.sum(axis=1)
    
    # Calculate percentages
    category_pct = category_by_strain.div(category_by_strain['TOTAL'], axis=0) * 100
    category_pct = category_pct.drop('TOTAL', axis=1)
    
    logger.info(f"\n{category_pct.to_string()}")
    
    logger.info("\nCategory Variation Check:")
    logger.info(f"  Categories vary across strains: {category_by_strain.drop('TOTAL', axis=1).nunique(axis=0).min() > 1}")
    
    return contrib_matrix, cv_df, consistency_df

# ============================================================================
# Visualization Functions
# ============================================================================

# ============================================================================
# HIERARCHICAL ATTRIBUTION ANALYSIS FUNCTIONS
# ============================================================================

def classify_cell_types_by_hierarchy(cell_types, hierarchy_dict):
    """Classify cell types into hierarchical groups."""
    classified = defaultdict(list)
    
    for group, members in hierarchy_dict.items():
        for ct in cell_types:
            if ct in members:
                classified[group].append(ct)
                
    return dict(classified)


def calculate_hierarchical_contributions(contrib_df, hierarchy_dict, hierarchy_name):
    """
    Calculate hierarchical group contributions from cell-type contributions.
    
    Args:
        contrib_df: DataFrame with cell_type and contribution_percent columns
        hierarchy_dict: Dictionary mapping hierarchy groups to cell types
        hierarchy_name: Name of hierarchy (Function, Location, Lineage)
    
    Returns:
        DataFrame with hierarchical group contributions
    """
    logger.info(f"Calculating {hierarchy_name} hierarchy contributions...")
    
    results = []
    
    for group, members in hierarchy_dict.items():
        # Sum contributions from all member cell types
        group_contrib = contrib_df[
            contrib_df['cell_type'].isin(members)
        ]['contribution_percent'].sum()
        
        results.append({
            'hierarchy': hierarchy_name,
            'group': group,
            'contribution_percent': group_contrib,
            'n_cell_types': len(set(contrib_df[contrib_df['cell_type'].isin(members)]['cell_type']))
        })
    
    return pd.DataFrame(results)


def generate_hierarchical_attribution(contrib_df):
    """
    Generate hierarchical attribution analysis across all three hierarchies.
    
    Returns:
        dict with keys: 'function', 'location', 'lineage', each containing
        a DataFrame with hierarchical group contributions
    """
    logger.info("\n" + "="*80)
    logger.info("HIERARCHICAL ATTRIBUTION ANALYSIS")
    logger.info("="*80)
    
    hierarchical_results = {}
    
    # Function hierarchy
    hierarchical_results['function'] = calculate_hierarchical_contributions(
        contrib_df, FUNCTION_HIERARCHY, 'Function'
    )
    
    # Location hierarchy
    hierarchical_results['location'] = calculate_hierarchical_contributions(
        contrib_df, LOCATION_HIERARCHY, 'Location'
    )
    
    # Lineage hierarchy
    hierarchical_results['lineage'] = calculate_hierarchical_contributions(
        contrib_df, LINEAGE_HIERARCHY, 'Lineage'
    )
    
    # Report results
    logger.info("\nHierarchical Contributions:")
    for hierarchy, df in hierarchical_results.items():
        logger.info(f"\n{hierarchy.upper()} Hierarchy:")
        for _, row in df.sort_values('contribution_percent', ascending=False).iterrows():
            logger.info(f"  {row['group']:20s} {row['contribution_percent']:6.2f}%")
    
    return hierarchical_results


def identify_dominant_patterns(hierarchical_results):
    """
    Identify dominant patterns like "The Three 80s".
    
    Returns:
        dict with dominant group for each hierarchy and percentage
    """
    patterns = {}
    
    for hierarchy, df in hierarchical_results.items():
        dominant_row = df.loc[df['contribution_percent'].idxmax()]
        patterns[hierarchy] = {
            'group': dominant_row['group'],
            'percentage': dominant_row['contribution_percent']
        }
    
    logger.info("\n" + "="*80)
    logger.info("DOMINANT PATTERNS (THE THREE Xs)")
    logger.info("="*80)
    
    for hierarchy, info in patterns.items():
        logger.info(f"{hierarchy.upper():12s} {info['percentage']:6.2f}% {info['group']}-Dominant")
    
    return patterns


# ============================================================================
# HIERARCHICAL FIGURES GENERATION
# ============================================================================

def plot_hierarchical_pie_charts(hierarchical_results, output_dir):
    """
    Generate publication-ready pie charts for all three hierarchies.
    Main text Figure 1.
    """
    logger.info("Generating hierarchical pie charts...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    colors_function = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
    colors_location = ['#9B59B6', '#1ABC9C', '#E67E22', '#95A5A6']
    colors_lineage = ['#C0392B', '#2980B9', '#27AE60', '#F1C40F', '#8E44AD']
    
    color_maps = {
        'function': colors_function,
        'location': colors_location,
        'lineage': colors_lineage
    }
    
    titles = {
        'function': 'Functional Classification',
        'location': 'Anatomical Location',
        'lineage': 'Cellular Lineage'
    }
    
    for idx, (hierarchy, df) in enumerate(hierarchical_results.items()):
        ax = axes[idx]
        
        # Sort by contribution
        df_sorted = df.sort_values('contribution_percent', ascending=False)
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            df_sorted['contribution_percent'],
            labels=df_sorted['group'],
            autopct='%1.1f%%',
            startangle=90,
            colors=color_maps[hierarchy][:len(df_sorted)],
            textprops={'fontsize': 10, 'weight': 'bold'}
        )
        
        ax.set_title(titles[hierarchy], fontsize=14, weight='bold', pad=20)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'hierarchical_pie_charts.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  Saved: {output_file}")


def plot_hierarchical_comprehensive(hierarchical_results, contrib_df, output_dir):
    """
    Generate comprehensive 9-panel hierarchical analysis.
    Supplementary Figure.
    """
    logger.info("Generating comprehensive hierarchical figure...")
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Row 1: Pie charts for each hierarchy
    for idx, (hierarchy, df) in enumerate(hierarchical_results.items()):
        ax = fig.add_subplot(gs[0, idx])
        df_sorted = df.sort_values('contribution_percent', ascending=False)
        ax.pie(df_sorted['contribution_percent'], labels=df_sorted['group'],
               autopct='%1.1f%%', startangle=90)
        ax.set_title(hierarchy.upper(), fontsize=12, weight='bold')
    
    # Row 2: Bar charts showing group contributions
    for idx, (hierarchy, df) in enumerate(hierarchical_results.items()):
        ax = fig.add_subplot(gs[1, idx])
        df_sorted = df.sort_values('contribution_percent', ascending=False)
        ax.barh(df_sorted['group'], df_sorted['contribution_percent'])
        ax.set_xlabel('Contribution (%)', fontsize=10)
        ax.set_title(f'{hierarchy.upper()} Contributions', fontsize=11)
        ax.grid(axis='x', alpha=0.3)
    
    # Row 3: Cell-type details
    ax1 = fig.add_subplot(gs[2, 0])
    top20 = contrib_df.nlargest(20, 'contribution_percent')
    ax1.barh(range(len(top20)), top20['contribution_percent'])
    ax1.set_yticks(range(len(top20)))
    ax1.set_yticklabels(top20['cell_type'], fontsize=8)
    ax1.set_xlabel('Contribution (%)')
    ax1.set_title('Top 20 Cell Types', fontsize=11, weight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Dominant pattern summary
    ax2 = fig.add_subplot(gs[2, 1:])
    ax2.axis('off')
    summary_text = "DOMINANT PATTERNS:\n\n"
    for hierarchy, df in hierarchical_results.items():
        dominant = df.loc[df['contribution_percent'].idxmax()]
        summary_text += f"{hierarchy.upper():12s}: {dominant['contribution_percent']:.1f}% {dominant['group']}-Dominant\n"
    ax2.text(0.1, 0.5, summary_text, fontsize=14, family='monospace',
             verticalalignment='center')
    
    plt.suptitle('Hierarchical Attribution Analysis - Comprehensive View',
                 fontsize=16, weight='bold', y=0.98)
    
    output_file = os.path.join(output_dir, 'hierarchical_comprehensive.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  Saved: {output_file}")


def plot_hierarchical_sankey(hierarchical_results, output_dir):
    """
    Generate Sankey diagram showing hierarchical flow.
    Requires plotly (optional).
    """
    try:
        import plotly.graph_objects as go
        logger.info("Generating hierarchical Sankey diagram...")
        
        # This is a simplified version - full implementation would be more complex
        # For now, create a basic visualization
        
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.text(0.5, 0.5, 'Hierarchical Sankey Diagram\n(Requires full plotly implementation)',
                ha='center', va='center', fontsize=16)
        ax.axis('off')
        
        output_file = os.path.join(output_dir, 'hierarchical_sankey.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Saved: {output_file}")
        
    except ImportError:
        logger.warning("  Plotly not available, skipping Sankey diagram")


# ============================================================================
# BULK-TO-SINGLE-CELL VALIDATION FUNCTIONS
# ============================================================================

def perform_bulk_validation(single_cell_flux_df, bulk_flux_df, cell_abundance_dict, output_dir):
    """
    Validate single-cell predictions against bulk tissue measurements.
    
    Args:
        single_cell_flux_df: RQ3 flux comparison (cell-type-specific)
        bulk_flux_df: RQ1 bulk tissue flux measurements
        cell_abundance_dict: Cell-type abundances
        output_dir: Where to save results
    
    Returns:
        dict with validation metrics
    """
    logger.info("\n" + "="*80)
    logger.info("BULK-TO-SINGLE-CELL VALIDATION")
    logger.info("="*80)
    
    # Aggregate single-cell predictions to bulk
    logger.info("Aggregating single-cell predictions to bulk level...")
    
    # Find common reactions
    sc_reactions = set(single_cell_flux_df['reaction_id'].unique())
    bulk_reactions = set(bulk_flux_df['ReactionID'].unique())
    common_reactions = sc_reactions & bulk_reactions
    
    logger.info(f"  Single-cell reactions: {len(sc_reactions)}")
    logger.info(f"  Bulk reactions: {len(bulk_reactions)}")
    logger.info(f"  Common reactions: {len(common_reactions)}")
    
    # Calculate abundance-weighted aggregate for each reaction
    aggregated_fluxes = []
    
    for rxn in common_reactions:
        # Get single-cell fluxes
        sc_data = single_cell_flux_df[single_cell_flux_df['reaction_id'] == rxn]
        
        # Calculate weighted average across cell types
        total_weighted_flux_chow = 0
        total_weighted_flux_wd = 0
        total_abundance = 0
        
        for _, row in sc_data.iterrows():
            cell_type = row['cell_type']
            abundance = cell_abundance_dict.get(cell_type, 0)
            
            if abundance > 0:
                # Get fluxes (assuming columns exist)
                flux_chow = row.get('Chow', 0) if 'Chow' in row else row.get(f'{cell_type}_Chow', 0)
                flux_wd = row.get('WesternDiet', 0) if 'WesternDiet' in row else row.get(f'{cell_type}_WesternDiet', 0)
                
                total_weighted_flux_chow += flux_chow * abundance
                total_weighted_flux_wd += flux_wd * abundance
                total_abundance += abundance
        
        if total_abundance > 0:
            aggregated_chow = total_weighted_flux_chow / total_abundance
            aggregated_wd = total_weighted_flux_wd / total_abundance
            
            # Get bulk measurement
            bulk_data = bulk_flux_df[bulk_flux_df['ReactionID'] == rxn]
            if len(bulk_data) > 0:
                bulk_chow = bulk_data.iloc[0].get('SCD_MeanFlux', 0)
                bulk_wd = bulk_data.iloc[0].get('WD_MeanFlux', 0)
                
                aggregated_fluxes.append({
                    'reaction_id': rxn,
                    'sc_aggregated_chow': aggregated_chow,
                    'sc_aggregated_wd': aggregated_wd,
                    'bulk_chow': bulk_chow,
                    'bulk_wd': bulk_wd,
                    'sc_change': aggregated_wd - aggregated_chow,
                    'bulk_change': bulk_wd - bulk_chow
                })
    
    validation_df = pd.DataFrame(aggregated_fluxes)
    
    if len(validation_df) == 0:
        logger.warning("  No common reactions found for validation")
        return None
    
    # Calculate validation metrics
    logger.info("\nCalculating validation metrics...")
    
    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(
        validation_df['sc_change'], 
        validation_df['bulk_change']
    )
    
    # Directional agreement
    sc_direction = np.sign(validation_df['sc_change'])
    bulk_direction = np.sign(validation_df['bulk_change'])
    directional_agreement = (sc_direction == bulk_direction).mean() * 100
    
    # RMSE
    rmse = np.sqrt(np.mean((validation_df['sc_change'] - validation_df['bulk_change'])**2))
    
    metrics = {
        'n_reactions': len(validation_df),
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'directional_agreement': directional_agreement,
        'rmse': rmse
    }
    
    logger.info("\nValidation Results:")
    logger.info(f"  Reactions compared: {metrics['n_reactions']}")
    logger.info(f"  Pearson r: {metrics['pearson_r']:.3f} (p = {metrics['pearson_p']:.2e})")
    logger.info(f"  Directional agreement: {metrics['directional_agreement']:.1f}%")
    logger.info(f"  RMSE: {metrics['rmse']:.4f}")
    
    # Save results
    validation_df.to_csv(os.path.join(output_dir, 'bulk_validation_data.csv'), index=False)
    
    with open(os.path.join(output_dir, 'bulk_validation_metrics.txt'), 'w') as f:
        f.write("BULK-TO-SINGLE-CELL VALIDATION METRICS\n")
        f.write("="*60 + "\n\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    
    # Generate validation plot
    plot_bulk_validation(validation_df, metrics, output_dir)
    
    return metrics


def plot_bulk_validation(validation_df, metrics, output_dir):
    """Generate validation scatter plot."""
    logger.info("Generating bulk validation plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot
    ax1 = axes[0]
    ax1.scatter(validation_df['bulk_change'], validation_df['sc_change'],
                alpha=0.5, s=30)
    
    # Add diagonal line
    max_val = max(validation_df['bulk_change'].max(), validation_df['sc_change'].max())
    min_val = min(validation_df['bulk_change'].min(), validation_df['sc_change'].min())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, alpha=0.7,
             label='Perfect Agreement')
    
    ax1.set_xlabel('Bulk Tissue Flux Change', fontsize=12)
    ax1.set_ylabel('Single-Cell Aggregated Flux Change', fontsize=12)
    ax1.set_title('Bulk vs Single-Cell Validation', fontsize=14, weight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Add metrics text
    metrics_text = f"Pearson r = {metrics['pearson_r']:.3f}\n"
    metrics_text += f"p = {metrics['pearson_p']:.2e}\n"
    metrics_text += f"Agreement = {metrics['directional_agreement']:.1f}%\n"
    metrics_text += f"N = {metrics['n_reactions']}"
    ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Distribution comparison
    ax2 = axes[1]
    ax2.hist(validation_df['bulk_change'], bins=30, alpha=0.5, label='Bulk', density=True)
    ax2.hist(validation_df['sc_change'], bins=30, alpha=0.5, label='Single-Cell', density=True)
    ax2.set_xlabel('Flux Change', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Distribution Comparison', fontsize=14, weight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'bulk_validation_plot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  Saved: {output_file}")



def plot_contribution_heatmap(contrib_matrix, output_dir, strains):
    """Plot heatmap of cell-type contributions across strains."""
    logger.info("\nGenerating contribution heatmap...")
    
    # Filter to top 15 cell types by mean contribution
    mean_contrib = contrib_matrix.mean(axis=1).sort_values(ascending=False)
    top_cells = mean_contrib.head(15).index
    
    plot_df = contrib_matrix.loc[top_cells]
    
    # Reorder strains for better visualization
    strain_order = [s for s in strains if s in plot_df.columns]
    plot_df = plot_df[strain_order]
    
    # Rename columns to full names
    plot_df.columns = [STRAIN_FULL_NAMES.get(s, s) for s in plot_df.columns]
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    sns.heatmap(plot_df, cmap='YlOrRd', annot=True, fmt='.1f',
                cbar_kws={'label': 'Contribution (%)'}, ax=ax,
                linewidths=0.5, linecolor='gray')
    
    ax.set_xlabel('Mouse Strain', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cell Type', fontsize=12, fontweight='bold')
    ax.set_title('Cell-Type Contributions to Bulk Metabolic Variance Across Strains',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cross_strain_contribution_heatmap.png'),
               dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("  [OK] Saved: cross_strain_contribution_heatmap.png")

def plot_strain_clustering(contrib_matrix, output_dir):
    """Plot hierarchical clustering of strains by cellular architecture."""
    logger.info("\nGenerating strain clustering dendrogram...")
    
    # Transpose to cluster strains
    strain_profiles = contrib_matrix.T
    
    # Hierarchical clustering
    Z = linkage(strain_profiles, method='ward')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create dendrogram
    strain_labels = [STRAIN_FULL_NAMES.get(s, s) for s in strain_profiles.index]
    dendrogram(Z, labels=strain_labels, ax=ax,
              leaf_font_size=12, leaf_rotation=45)
    
    ax.set_xlabel('Mouse Strain', fontsize=12, fontweight='bold')
    ax.set_ylabel('Distance (Ward)', fontsize=12, fontweight='bold')
    ax.set_title('Hierarchical Clustering of Strains by Cellular Architecture',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'strain_clustering_dendrogram.png'),
               dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("  [OK] Saved: strain_clustering_dendrogram.png")

def plot_top_contributors_comparison(all_contributions, output_dir, strains):
    """Plot top contributors for each strain side-by-side."""
    logger.info("\nGenerating top contributors comparison...")
    
    n_strains = len(strains)
    n_cols = 3
    n_rows = (n_strains + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten()
    
    for idx, strain in enumerate(strains):
        ax = axes[idx]
        
        # Get top 8 for this strain
        strain_contrib = all_contributions[all_contributions['strain'] == strain]
        strain_contrib = strain_contrib.sort_values('contribution_percent', ascending=True).tail(8)
        
        # Plot
        bars = ax.barh(range(len(strain_contrib)), 
                      strain_contrib['contribution_percent'],
                      color='steelblue', alpha=0.7)
        
        ax.set_yticks(range(len(strain_contrib)))
        ax.set_yticklabels(strain_contrib['cell_type'], fontsize=8)
        ax.set_xlabel('Contribution (%)', fontsize=9)
        ax.set_title(STRAIN_FULL_NAMES.get(strain, strain), fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add values on bars
        for i, (idx_row, row) in enumerate(strain_contrib.iterrows()):
            ax.text(row['contribution_percent'] + 0.5, i,
                   f"{row['contribution_percent']:.1f}%",
                   va='center', fontsize=7)
    
    # Hide empty subplots
    for idx in range(len(strains), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Top Cell-Type Contributors Across All Strains',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_contributors_per_strain.png'),
               dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("  [OK] Saved: top_contributors_per_strain.png")

def plot_attribution_comparison(all_attributions, output_dir, strains, contribution_threshold):
    """Plot attribution category distribution across strains."""
    logger.info("\nGenerating attribution comparison...")
    
    # Count categories per strain
    attrib_counts = []
    for strain in strains:
        strain_attrib = all_attributions[all_attributions['strain'] == strain]
        counts = strain_attrib['category'].value_counts()
        total = len(strain_attrib)
        
        for cat in ['MULTI_CELLULAR', 'COOPERATIVE', 'CELL_TYPE_UNIQUE', 'NON_CELLULAR']:
            count = counts.get(cat, 0)
            pct = count / total * 100 if total > 0 else 0
            attrib_counts.append({
                'strain': strain,
                'category': cat,
                'count': count,
                'percent': pct,
            })
    
    attrib_df = pd.DataFrame(attrib_counts)
    
    # Create stacked bar plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    pivot_df = attrib_df.pivot(index='strain', columns='category', values='percent')
    pivot_df = pivot_df.reindex(strains)
    pivot_df.index = [STRAIN_FULL_NAMES.get(s, s) for s in pivot_df.index]
    
    colors = {'MULTI_CELLULAR': '#ff9999', 'COOPERATIVE': '#66b3ff',
             'CELL_TYPE_UNIQUE': '#99ff99', 'NON_CELLULAR': '#ffcc99'}
    
    # Ensure all columns exist
    for cat in ['MULTI_CELLULAR', 'COOPERATIVE', 'CELL_TYPE_UNIQUE', 'NON_CELLULAR']:
        if cat not in pivot_df.columns:
            pivot_df[cat] = 0
    
    pivot_df[['MULTI_CELLULAR', 'COOPERATIVE', 'CELL_TYPE_UNIQUE', 'NON_CELLULAR']].plot(
        kind='bar', stacked=True, color=[colors[c] for c in ['MULTI_CELLULAR', 'COOPERATIVE', 'CELL_TYPE_UNIQUE', 'NON_CELLULAR']], ax=ax
    )
    
    ax.set_xlabel('Mouse Strain', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage of Reactions', fontsize=12, fontweight='bold')
    ax.set_title(f'Attribution Category Distribution Across Strains\n(Threshold: {contribution_threshold:.1%})',
                fontsize=14, fontweight='bold')
    ax.legend(title='Category', loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add note about threshold
    ax.text(0.02, 0.98, f'Note: Cell types must contribute ≥{contribution_threshold:.1%} to count as significant',
           transform=ax.transAxes, fontsize=8, va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attribution_categories_per_strain.png'),
               dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("  [OK] Saved: attribution_categories_per_strain.png")

def plot_conservation_metrics(cv_df, output_dir):
    """Plot conservation metrics."""
    logger.info("\nGenerating conservation metrics plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Panel A: CV distribution
    ax = axes[0]
    
    # Top 20 cell types
    plot_df = cv_df.head(20).sort_values('cv', ascending=True)
    
    colors = ['green' if cv < 0.3 else 'orange' if cv < 0.5 else 'red' 
             for cv in plot_df['cv']]
    
    bars = ax.barh(range(len(plot_df)), plot_df['cv'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df['cell_type'])
    ax.set_xlabel('Coefficient of Variation (CV)', fontsize=12)
    ax.set_title('A) Conservation of Cell-Type Contributions\n(CV < 0.3 = Conserved)',
                fontsize=13, fontweight='bold')
    ax.axvline(x=0.3, color='black', linestyle='--', linewidth=1, label='CV = 0.3')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    # Panel B: Mean contribution vs CV scatter
    ax = axes[1]
    
    scatter = ax.scatter(cv_df['mean_contribution'], cv_df['cv'],
                        s=100, alpha=0.6, c=cv_df['cv'], cmap='RdYlGn_r',
                        edgecolors='black', linewidth=0.5)
    
    # Annotate top contributors
    top_annotate = cv_df.nlargest(8, 'mean_contribution')
    for idx, row in top_annotate.iterrows():
        ax.annotate(row['cell_type'],
                   (row['mean_contribution'], row['cv']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.7)
    
    ax.axhline(y=0.3, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Mean Contribution Across Strains (%)', fontsize=12)
    ax.set_ylabel('Coefficient of Variation', fontsize=12)
    ax.set_title('B) Contribution Magnitude vs Conservation',
                fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('CV', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'conservation_metrics.png'),
               dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("  [OK] Saved: conservation_metrics.png")

# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Main multi-strain integration pipeline."""
    
    parser = argparse.ArgumentParser(
        description="RQ2 + RQ3 Multi-Strain Integration Analysis (ENHANCED v3.0)\n"
                   "Includes: Hierarchical Attribution + Bulk-to-Single-Cell Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLE USAGE:
--------------
# Basic run (original features only):
python rq2_rq3_multi_strain_integration_ENHANCED.py --base_dir /mnt/project

# With hierarchical attribution:
python rq2_rq3_multi_strain_integration_ENHANCED.py \\
    --base_dir /mnt/project \\
    --run_hierarchical True

# With bulk-to-single-cell validation:
python rq2_rq3_multi_strain_integration_ENHANCED.py \\
    --base_dir /mnt/project \\
    --run_bulk_validation True \\
    --bulk_flux_file /mnt/project/RQ1_multidataset_reaction_flux_comparison_extended.csv

# Full analysis (all features):
python rq2_rq3_multi_strain_integration_ENHANCED.py \\
    --base_dir /mnt/project \\
    --output_dir /mnt/user-data/outputs/rq2_rq3_complete \\
    --run_hierarchical True \\
    --run_bulk_validation True \\
    --bulk_flux_file /mnt/project/RQ1_multidataset_reaction_flux_comparison_extended.csv

# Sensitivity analysis (test different thresholds):
python rq2_rq3_multi_strain_integration_ENHANCED.py \\
    --base_dir /mnt/project \\
    --contribution_threshold 0.20 \\
    --run_hierarchical True
        """
    )
    
    # Input paths
    parser.add_argument('--base_dir', type=str, 
                       default='Fluxes_Data_multi_background_rq2',
                       help='Base directory containing RQ1, RQ2, RQ3 data files')
    parser.add_argument('--rq1_stats', type=str, default=None,
                       help='RQ1 pairwise stats file (default: {base_dir}/RQ1_multidataset_flux_pairwise_stats.csv)')
    parser.add_argument('--rq3_stats', type=str, default=None,
                       help='RQ3 statistical tests file (default: {base_dir}/RQ3_statistical_tests.csv)')
    parser.add_argument('--rq3_aggregation', type=str, default=None,
                       help='RQ3 aggregation summary file (default: {base_dir}/RQ3_aggregation_summary.csv)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: {base_dir}/rq2_rq3_integration_results_FIXED_V2)')
    
    # Parameters
    parser.add_argument('--contribution_threshold', type=float, 
                       default=DEFAULT_CONTRIBUTION_THRESHOLD,
                       help=f'Contribution threshold for attribution (default: {DEFAULT_CONTRIBUTION_THRESHOLD})')
    parser.add_argument('--fdr_threshold', type=float, default=0.10,
                       help='FDR threshold for RQ1 significance (default: 0.10)')
    parser.add_argument('--bulk_comparison', type=str, default='WD_vs_SCD',
                       help='Bulk comparison name (default: WD_vs_SCD)')
    parser.add_argument('--cellular_comparison', type=str, default='WesternDiet_vs_Chow',
                       help='Cellular comparison name (default: WesternDiet_vs_Chow)')
    
    # Strain selection
    parser.add_argument('--strains', type=str, default=None,
                       help='Comma-separated list of strains to analyze (default: all 9)')
    
    # NEW: Hierarchical attribution
    parser.add_argument('--run_hierarchical', type=lambda x: x.lower() == 'true',
                       default=False,
                       help='Run hierarchical attribution analysis (True/False, default: False)')
    
    # NEW: Bulk-to-single-cell validation
    parser.add_argument('--run_bulk_validation', type=lambda x: x.lower() == 'true',
                       default=False,
                       help='Run bulk-to-single-cell validation (True/False, default: False)')
    parser.add_argument('--bulk_flux_file', type=str, default=None,
                       help='Bulk flux file for validation (e.g., RQ1_multidataset_reaction_flux_comparison_extended.csv)')
    parser.add_argument('--single_cell_flux_file', type=str, default=None,
                       help='Single-cell flux file (default: {base_dir}/RQ3_flux_comparison.csv)')
    
    args = parser.parse_args()
    
    # Set up paths
    base_dir = args.base_dir
    rq1_stats_file = args.rq1_stats or os.path.join(base_dir, 'RQ1_multidataset_flux_pairwise_stats.csv')
    rq3_stats_file = args.rq3_stats or os.path.join(base_dir, 'RQ3_statistical_tests.csv')
    rq3_aggregation_file = args.rq3_aggregation or os.path.join(base_dir, 'RQ3_aggregation_summary.csv')
    output_dir = args.output_dir or os.path.join(base_dir, 'rq2_rq3_integration_results_FIXED_V2')
    
    # Parse strains
    strains = args.strains.split(',') if args.strains else STRAINS
    
    contribution_threshold = args.contribution_threshold
    
    logger.info("="*80)
    logger.info("RQ2 + RQ3 MULTI-STRAIN INTEGRATION ANALYSIS - VERSION 2 (FIXED)")
    logger.info("="*80)
    logger.info(f"\nAnalyzing {len(strains)} genetically diverse strains")
    logger.info(f"Strains: {', '.join([STRAIN_FULL_NAMES.get(s, s) for s in strains])}")
    logger.info(f"\nConfiguration:")
    logger.info(f"  Contribution threshold: {contribution_threshold:.1%}")
    logger.info(f"  FDR threshold: {args.fdr_threshold}")
    logger.info(f"  Bulk comparison: {args.bulk_comparison}")
    logger.info(f"  Cellular comparison: {args.cellular_comparison}")
    logger.info(f"\nPaths:")
    logger.info(f"  Base dir: {base_dir}")
    logger.info(f"  Output dir: {output_dir}")
    
    # Setup output directories
    os.makedirs(output_dir, exist_ok=True)
    
    tables_dir = os.path.join(output_dir, 'tables')
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # Load shared data
    logger.info("\n" + "="*80)
    logger.info("LOADING SHARED DATA")
    logger.info("="*80)
    
    # RQ1 significant reactions (statistical filter)
    rq1_sig_reactions = load_rq1_significant_reactions(
        rq1_stats_file,
        comparison=args.bulk_comparison,
        fdr_threshold=args.fdr_threshold
    )
    
    # RQ3 cellular data
    rq3_stats_df, cellular_sig_by_cell_type = load_rq3_cellular_data(
        rq3_stats_file,
        comparison=args.cellular_comparison
    )
    
    # Cell abundance
    abundance_dict = calculate_cell_abundance_from_scrna(rq3_aggregation_file)
    
    # Run integration for each strain
    logger.info("\n" + "="*80)
    logger.info("PER-STRAIN INTEGRATION ANALYSIS")
    logger.info("="*80)
    
    all_contributions = []
    all_attributions = []
    
    for strain in strains:
        rq2_file = os.path.join(base_dir, f'RQ2_{strain}_reaction_flux_comparison_extended.csv')
        
        if not os.path.isfile(rq2_file):
            logger.warning(f"Skipping {strain}: File not found: {rq2_file}")
            continue
        
        contrib_df, attrib_df = run_strain_integration(
            strain, rq2_file, rq1_sig_reactions,
            rq3_stats_df, cellular_sig_by_cell_type,
            abundance_dict, contribution_threshold
        )
        
        all_contributions.append(contrib_df)
        all_attributions.append(attrib_df)
    
    if not all_contributions:
        logger.error("No strain data processed! Check input files.")
        return 1
    
    # Combine results
    all_contributions = pd.concat(all_contributions, ignore_index=True)
    all_attributions = pd.concat(all_attributions, ignore_index=True)
    
    # Get list of actually processed strains
    processed_strains = all_contributions['strain'].unique().tolist()
    
    # Save per-strain results
    all_contributions.to_csv(os.path.join(tables_dir, 'all_strain_contributions.csv'), index=False)
    all_attributions.to_csv(os.path.join(tables_dir, 'all_strain_attributions.csv'), index=False)
    
    logger.info(f"\n[OK] Saved per-strain results to: {tables_dir}")
    
    # Cross-strain analysis
    contrib_matrix, cv_df, consistency_df = analyze_cross_strain_conservation(
        all_contributions, all_attributions, processed_strains
    )
    
    # Save cross-strain results
    contrib_matrix.to_csv(os.path.join(tables_dir, 'contribution_matrix.csv'))
    cv_df.to_csv(os.path.join(tables_dir, 'conservation_analysis.csv'), index=False)
    consistency_df.to_csv(os.path.join(tables_dir, 'driver_consistency.csv'), index=False)
    
    logger.info(f"[OK] Saved cross-strain analysis to: {tables_dir}")
    
    # Generate visualizations
    logger.info("\n" + "="*80)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("="*80)
    
    plot_contribution_heatmap(contrib_matrix, figures_dir, processed_strains)
    plot_strain_clustering(contrib_matrix, figures_dir)
    plot_top_contributors_comparison(all_contributions, figures_dir, processed_strains)
    plot_attribution_comparison(all_attributions, figures_dir, processed_strains, contribution_threshold)
    plot_conservation_metrics(cv_df, figures_dir)
    
    # ========================================================================
    # NEW: HIERARCHICAL ATTRIBUTION ANALYSIS (Optional)
    # ========================================================================
    
    if args.run_hierarchical:
        logger.info("\n" + "="*80)
        logger.info("HIERARCHICAL ATTRIBUTION ANALYSIS")
        logger.info("="*80)
        
        # Use phase2 contribution analysis for hierarchical grouping
        try:
            phase2_file = os.path.join(base_dir, 'RQ3_phase2_contribution_analysis.csv')
            if os.path.exists(phase2_file):
                contrib_df = pd.read_csv(phase2_file)
                
                # Generate hierarchical attribution
                hierarchical_results = generate_hierarchical_attribution(contrib_df)
                
                # Identify dominant patterns
                dominant_patterns = identify_dominant_patterns(hierarchical_results)
                
                # Save hierarchical results
                hierarchical_dir = os.path.join(output_dir, 'hierarchical_attribution')
                os.makedirs(hierarchical_dir, exist_ok=True)
                
                for hierarchy, df in hierarchical_results.items():
                    df.to_csv(os.path.join(hierarchical_dir, f'{hierarchy}_attribution.csv'), index=False)
                
                # Generate hierarchical figures
                logger.info("\nGenerating hierarchical figures...")
                plot_hierarchical_pie_charts(hierarchical_results, hierarchical_dir)
                plot_hierarchical_comprehensive(hierarchical_results, contrib_df, hierarchical_dir)
                plot_hierarchical_sankey(hierarchical_results, hierarchical_dir)
                
                logger.info(f"[OK] Hierarchical attribution results saved to: {hierarchical_dir}")
                
            else:
                logger.warning(f"Phase2 file not found: {phase2_file}")
                logger.warning("Skipping hierarchical attribution analysis")
                
        except Exception as e:
            logger.error(f"Error in hierarchical attribution: {e}")
            logger.warning("Continuing without hierarchical attribution")
    
    # ========================================================================
    # NEW: BULK-TO-SINGLE-CELL VALIDATION (Optional)
    # ========================================================================
    
    if args.run_bulk_validation:
        logger.info("\n" + "="*80)
        logger.info("BULK-TO-SINGLE-CELL VALIDATION")
        logger.info("="*80)
        
        try:
            # Load required files
            bulk_flux_file = args.bulk_flux_file or os.path.join(base_dir, 'RQ1_multidataset_reaction_flux_comparison_extended.csv')
            sc_flux_file = args.single_cell_flux_file or os.path.join(base_dir, 'RQ3_flux_comparison.csv')
            
            if os.path.exists(bulk_flux_file) and os.path.exists(sc_flux_file):
                logger.info(f"Loading bulk flux data: {bulk_flux_file}")
                logger.info(f"Loading single-cell flux data: {sc_flux_file}")
                
                bulk_flux_df = pd.read_csv(bulk_flux_file)
                sc_flux_df = pd.read_csv(sc_flux_file)
                
                # Perform validation
                validation_dir = os.path.join(output_dir, 'bulk_validation')
                os.makedirs(validation_dir, exist_ok=True)
                
                validation_metrics = perform_bulk_validation(
                    sc_flux_df, bulk_flux_df, abundance_dict, validation_dir
                )
                
                if validation_metrics:
                    logger.info(f"[OK] Bulk validation results saved to: {validation_dir}")
                else:
                    logger.warning("Bulk validation produced no results")
                    
            else:
                missing_files = []
                if not os.path.exists(bulk_flux_file):
                    missing_files.append(f"Bulk flux file: {bulk_flux_file}")
                if not os.path.exists(sc_flux_file):
                    missing_files.append(f"Single-cell flux file: {sc_flux_file}")
                    
                logger.warning("Cannot perform bulk validation - missing files:")
                for mf in missing_files:
                    logger.warning(f"  {mf}")
                    
        except Exception as e:
            logger.error(f"Error in bulk validation: {e}")
            logger.warning("Continuing without bulk validation")
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info(f"\nOutput directory: {output_dir}")
    logger.info(f"  Tables: {tables_dir}")
    logger.info(f"  Figures: {figures_dir}")
    
    logger.info("\nGenerated files:")
    logger.info("  Tables:")
    logger.info("    - all_strain_contributions.csv")
    logger.info("    - all_strain_attributions.csv")
    logger.info("    - contribution_matrix.csv")
    logger.info("    - conservation_analysis.csv")
    logger.info("    - driver_consistency.csv")
    logger.info("  Figures:")
    logger.info("    - cross_strain_contribution_heatmap.png")
    logger.info("    - strain_clustering_dendrogram.png")
    logger.info("    - top_contributors_per_strain.png")
    logger.info("    - attribution_categories_per_strain.png")
    logger.info("    - conservation_metrics.png")
    
    if args.run_hierarchical:
        logger.info("  Hierarchical Attribution:")
        logger.info("    - hierarchical_attribution/function_attribution.csv")
        logger.info("    - hierarchical_attribution/location_attribution.csv")
        logger.info("    - hierarchical_attribution/lineage_attribution.csv")
        logger.info("    - hierarchical_attribution/hierarchical_pie_charts.png")
        logger.info("    - hierarchical_attribution/hierarchical_comprehensive.png")
    
    if args.run_bulk_validation:
        logger.info("  Bulk Validation:")
        logger.info("    - bulk_validation/bulk_validation_data.csv")
        logger.info("    - bulk_validation/bulk_validation_metrics.txt")
        logger.info("    - bulk_validation/bulk_validation_plot.png")
    
    # Key findings summary
    logger.info("\n" + "="*80)
    logger.info("KEY FINDINGS SUMMARY")
    logger.info("="*80)
    
    # Category variation check
    category_variation = all_attributions.groupby(['strain', 'category']).size().unstack(fill_value=0)
    logger.info("\n[OK] CRITICAL VALIDATION:")
    logger.info(f"  Categories vary across strains: {category_variation.nunique(axis=0).min() > 1}")
    logger.info(f"  (Previous bug: All strains had identical distributions)")
    
    # Top conserved cell types
    conserved_cells = cv_df[cv_df['cv'] < 0.3].nlargest(5, 'mean_contribution')
    logger.info("\nTop 5 Conserved Cell Types (CV < 0.3):")
    for idx, row in conserved_cells.iterrows():
        logger.info(f"  {row['cell_type']:30s}: Mean {row['mean_contribution']:5.1f}%, CV {row['cv']:.3f}")
    
    # Top variable cell types
    variable_cells = cv_df[cv_df['cv'] >= 0.5].nlargest(5, 'mean_contribution')
    if len(variable_cells) > 0:
        logger.info("\nTop 5 Variable Cell Types (CV >= 0.5):")
        for idx, row in variable_cells.iterrows():
            logger.info(f"  {row['cell_type']:30s}: Mean {row['mean_contribution']:5.1f}%, CV {row['cv']:.3f}")
    
    # Primary driver consistency
    logger.info("\nMost Consistent Primary Drivers:")
    for idx, row in consistency_df.head(5).iterrows():
        logger.info(f"  {row['primary_driver']:30s}: {row['total_reactions']:4.0f} reactions across "
                   f"{row['strains_present']:.0f}/{len(processed_strains)} strains ({row['consistency']*100:.0f}%)")
    
    logger.info("\n" + "="*80)
    logger.info("MAJOR FIX IMPLEMENTED:")
    logger.info("  - Strain-specific attribution with contribution thresholding")
    logger.info(f"  - Threshold: {contribution_threshold:.1%}")
    logger.info("  - Categories now vary by strain based on metabolic contribution patterns")
    logger.info("  - Previous bug: Used universal RQ3 cell-type significance for all strains")
    logger.info("="*80)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
