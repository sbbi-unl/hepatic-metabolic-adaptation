#!/usr/bin/env python3
"""
===============================================================================
RQ1-RQ2-RQ3 INTEGRATION ANALYSIS (REVISED)
===============================================================================
Links bulk-liver metabolic rewiring (RQ1/RQ2) to cell-type-specific 
responses (RQ3) to identify which hepatic cell populations drive 
bulk-level changes.

REQUIRED INPUT FILES:
--------------------
1. RQ1 pairwise statistics CSV
   - File: RQ1_multidataset_flux_pairwise_stats.csv
   - Contains: Bulk-level statistical tests across diets
   - Columns: ReactionID, ReactionName, Subsystem, GroupA, GroupB, 
             MeanA, MeanB, Diff(A-B), TestStat, PValue, FDR_BH

2. RQ3 statistical tests CSV
   - File: statistical_tests.csv (from RQ3 pipeline output)
   - Contains: Cell-type-specific statistical tests
   - Columns: reaction_id, reaction_name, subsystem, cell_type, comparison,
             baseline_flux, test_flux, flux_change, cohens_d, significant

3. RQ3 aggregation summary CSV
   - File: aggregation_summary.csv (from RQ3 pipeline output)
   - Contains: Cell counts per cell type for abundance calculation
   - Columns: cell_type, condition, n_cells, genes_detected, etc.
===============================================================================
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION: Cell Abundance Estimates
# ============================================================================

# Literature-based estimates of hepatic cell-type abundance
# Source: Ben-Moshe et al., Nature 2019; Halpern et al., Nature 2017
# Note: These are in situ estimates (tissue volume/cell counts)

LITERATURE_CELL_ABUNDANCE = {
    # Parenchymal cells (70-80% of liver)
    'Hepatocytes': 0.75,  # 70-80% - dominant cell type
    
    # Non-parenchymal cells (20-30% of liver)
    'LECs': 0.08,  # Liver endothelial cells (sinusoidal)
    'Kupffer': 0.08,  # Resident macrophages (combined Timd4+ and Cd207+)
    'Timd4+ resKC': 0.04,  # Tissue-resident Kupffer
    'Cd207+, Trem2+ Mo-KC': 0.02,  # Monocyte-derived Kupffer
    'Cd207-, Trem2+ Mo-KC': 0.02,
    
    # Stellate cells
    'qHSCs': 0.04,  # Quiescent hepatic stellate cells
    'aHSCs': 0.01,  # Activated hepatic stellate cells  
    'cAMP qHSCs': 0.01,
    
    # Other immune cells
    'Macrophages': 0.01,
    'Dendrites': 0.005,
    'T cells': 0.005,
    'B cells': 0.003,
    'Neutrophiles': 0.002,
    'Ly6c2+, Ccr2+ Mo': 0.001,
    'Ly6c2-, Spn+ Mo': 0.001,
    'Transitioning Mo': 0.001,
    'Cx3cr1+, Ccr2+ MdM': 0.002,
    'Trem2+, Spp1+ MdM': 0.002,
    'DCs': 0.001,
    
    # Other cell types
    'Cholangiocytes': 0.03,  # Bile duct epithelial cells
    'MCs': 0.005,  # Mesothelial cells
    'Mast cells': 0.001,
    'Cycling': 0.001,
}

def calculate_cell_abundance_from_scrna(aggregation_summary_file):
    """
    Calculate cell-type abundance from scRNA-seq data.
    
    Uses total cell counts across all conditions to estimate relative abundance.
    This is more accurate than literature estimates for this specific dataset.
    
    Parameters
    ----------
    aggregation_summary_file : str
        Path to aggregation_summary.csv from RQ3 pipeline
    
    Returns
    -------
    dict
        Dictionary mapping cell_type -> abundance fraction
    """
    logger.info(f"Calculating cell abundance from scRNA-seq data: {aggregation_summary_file}")
    
    # Load aggregation summary
    agg_df = pd.read_csv(aggregation_summary_file)
    
    # Sum cells per cell type across all conditions
    cell_counts = agg_df.groupby('cell_type')['n_cells'].sum()
    total_cells = cell_counts.sum()
    
    # Calculate fractional abundance
    abundance = (cell_counts / total_cells).to_dict()
    
    logger.info(f"  Total cells analyzed: {total_cells:,}")
    logger.info(f"  Cell types detected: {len(abundance)}")
    logger.info("\n  Top 10 most abundant cell types:")
    for ct, frac in sorted(abundance.items(), key=lambda x: x[1], reverse=True)[:10]:
        logger.info(f"    {ct:30s}: {frac*100:5.1f}% ({cell_counts[ct]:,} cells)")
    
    return abundance

def get_cell_abundance(cell_type, abundance_dict):
    """
    Get cell abundance for a given cell type.
    
    Parameters
    ----------
    cell_type : str
        Cell type name
    abundance_dict : dict
        Abundance dictionary (from scRNA-seq or literature)
    
    Returns
    -------
    float
        Abundance fraction (0-1)
    """
    return abundance_dict.get(cell_type, 0.001)  # Default 0.1% if unknown

# ============================================================================
# Phase 1: Bulk-Cellular Overlap Analysis
# ============================================================================

def load_rq1_bulk_significant_reactions(rq1_pairwise_stats_file, 
                                        comparison='WD_vs_SCD', 
                                        fdr_threshold=0.10):
    """
    Load bulk-significant reactions from RQ1 pairwise statistical tests.
    
    IMPORTANT: Diet naming convention
    -----------------------------------
    RQ1 uses: SCD, HFD, WD, KD
    RQ3 uses: Chow, WesternDiet
    
    MAPPING:
      SCD = Chow (Standard Chow Diet)
      WD  = WesternDiet (Western Diet)
      HFD = High-Fat Diet (different from WD!)
    
    To match RQ3's "WesternDiet_vs_Chow", use comparison='WD_vs_SCD'
    
    Parameters
    ----------
    rq1_pairwise_stats_file : str
        Path to RQ1 pairwise stats CSV
    comparison : str
        Which comparison to use (default: 'WD_vs_SCD' to match RQ3)
        Options: 'HFD_vs_SCD', 'WD_vs_SCD', 'KD_vs_SCD'
    fdr_threshold : float
        FDR threshold for significance (default: 0.10)
    
    Returns
    -------
    pd.DataFrame
        Bulk-significant reactions with statistical metrics
    """
    logger.info(f"Loading RQ1 bulk significant reactions from {rq1_pairwise_stats_file}")
    logger.info(f"  Comparison: {comparison} (bulk) to match cellular WesternDiet_vs_Chow")
    
    # Load pairwise stats
    stats_df = pd.read_csv(rq1_pairwise_stats_file)
    
    # Map comparison names
    comparison_map = {
        'HFD_vs_SCD': ('HFD', 'SCD'),
        'WD_vs_SCD': ('SCD', 'WD'),  # Note: order in file is reversed
        'KD_vs_SCD': ('KD', 'SCD'),
    }
    
    if comparison not in comparison_map:
        raise ValueError(f"Invalid comparison: {comparison}. Must be one of {list(comparison_map.keys())}")
    
    group_a, group_b = comparison_map[comparison]
    
    # Filter for this comparison
    mask = ((stats_df['GroupA'] == group_a) & (stats_df['GroupB'] == group_b))
    comp_df = stats_df[mask].copy()
    
    if len(comp_df) == 0:
        logger.warning(f"No data found for comparison {comparison} (GroupA={group_a}, GroupB={group_b})")
        logger.warning("Available comparisons in file:")
        available = stats_df.groupby(['GroupA', 'GroupB']).size()
        for (ga, gb), count in available.items():
            logger.warning(f"  {ga} vs {gb}: {count} reactions")
    
    # Filter for significant reactions (FDR < threshold)
    sig_mask = comp_df['FDR_BH'] < fdr_threshold
    bulk_sig = comp_df[sig_mask].copy()
    
    logger.info(f"  Total reactions tested: {len(comp_df)}")
    logger.info(f"  Significant (FDR < {fdr_threshold}): {len(bulk_sig)} ({len(bulk_sig)/len(comp_df)*100:.1f}%)")
    
    return bulk_sig

def load_rq3_cellular_significant_reactions(rq3_stats_file,
                                            comparison='WesternDiet_vs_Chow'):
    """
    Load cell-type-specific significant reactions from RQ3.
    
    Parameters
    ----------
    rq3_stats_file : str
        Path to RQ3 statistical tests CSV
    comparison : str
        Which comparison (default: 'WesternDiet_vs_Chow')
    
    Returns
    -------
    tuple
        (rq3_stats_df, sig_by_cell_type_dict)
    """
    logger.info(f"Loading RQ3 cellular significant reactions from {rq3_stats_file}")
    logger.info(f"  Comparison: {comparison} (cellular)")
    
    # Load RQ3 stats
    rq3_df = pd.read_csv(rq3_stats_file)
    
    # Filter for this comparison
    rq3_comp = rq3_df[rq3_df['comparison'] == comparison].copy()
    
    if len(rq3_comp) == 0:
        logger.warning(f"No data found for comparison {comparison}")
        logger.warning("Available comparisons in file:")
        available = rq3_df['comparison'].unique()
        for comp in available:
            logger.warning(f"  {comp}")
    
    # Get significant reactions per cell type
    sig_by_cell_type = {}
    for cell_type in rq3_comp['cell_type'].unique():
        ct_df = rq3_comp[rq3_comp['cell_type'] == cell_type]
        sig_rxns = ct_df[ct_df['significant'] == True]['reaction_id'].unique()
        sig_by_cell_type[cell_type] = set(sig_rxns)
        logger.info(f"  {cell_type:30s}: {len(sig_rxns):4d} significant reactions")
    
    return rq3_comp, sig_by_cell_type

def calculate_bulk_cellular_overlap(bulk_sig_reactions, cellular_sig_by_cell_type, abundance_dict):
    """
    Calculate overlap between bulk-significant and cell-type-significant reactions.
    
    Parameters
    ----------
    bulk_sig_reactions : pd.DataFrame
        Bulk-significant reactions
    cellular_sig_by_cell_type : dict
        Dictionary mapping cell_type -> set of significant reaction IDs
    abundance_dict : dict
        Cell abundance dictionary
    
    Returns
    -------
    pd.DataFrame
        Overlap analysis results
    """
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: BULK-CELLULAR OVERLAP ANALYSIS")
    logger.info("="*80)
    
    bulk_rxn_set = set(bulk_sig_reactions['ReactionID'])
    
    overlap_results = []
    
    for cell_type, cell_rxn_set in cellular_sig_by_cell_type.items():
        # Calculate overlap
        overlap = bulk_rxn_set & cell_rxn_set
        bulk_only = bulk_rxn_set - cell_rxn_set
        cell_only = cell_rxn_set - bulk_rxn_set
        
        # Overlap percentage (of bulk-significant reactions)
        overlap_pct = len(overlap) / len(bulk_rxn_set) * 100 if len(bulk_rxn_set) > 0 else 0
        
        # Sensitivity (recall): fraction of bulk reactions detected in this cell type
        sensitivity = len(overlap) / len(bulk_rxn_set) if len(bulk_rxn_set) > 0 else 0
        
        # Precision: fraction of cell-type reactions that are also bulk-significant
        precision = len(overlap) / len(cell_rxn_set) if len(cell_rxn_set) > 0 else 0
        
        # F1 score
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        overlap_results.append({
            'cell_type': cell_type,
            'n_bulk_sig': len(bulk_rxn_set),
            'n_cell_sig': len(cell_rxn_set),
            'n_overlap': len(overlap),
            'n_bulk_only': len(bulk_only),
            'n_cell_only': len(cell_only),
            'overlap_pct': overlap_pct,
            'sensitivity': sensitivity,
            'precision': precision,
            'f1_score': f1,
            'cell_abundance': get_cell_abundance(cell_type, abundance_dict),
        })
    
    overlap_df = pd.DataFrame(overlap_results)
    overlap_df = overlap_df.sort_values('overlap_pct', ascending=False)
    
    logger.info("\nTop 10 cell types by overlap with bulk-significant reactions:")
    logger.info("-" * 80)
    logger.info(f"{'Cell Type':<30s} {'Overlap':>8s} {'%':>6s} {'F1':>6s} {'Abundance':>10s}")
    logger.info("-" * 80)
    for idx, row in overlap_df.head(10).iterrows():
        logger.info(f"{row['cell_type']:<30s} {int(row['n_overlap']):>8d} {row['overlap_pct']:>6.1f} {row['f1_score']:>6.3f} {row['cell_abundance']*100:>9.1f}%")
    
    return overlap_df

# ============================================================================
# Phase 2: Abundance-Weighted Contribution Analysis
# ============================================================================

def calculate_abundance_weighted_contributions(cellular_sig_by_cell_type, 
                                              rq3_stats_df,
                                              abundance_dict,
                                              comparison='WesternDiet_vs_Chow'):
    """
    Calculate abundance-weighted contributions of each cell type to bulk metabolic variance.
    
    contribution_score = cell_abundance × response_rate × mean_flux_change
    
    Parameters
    ----------
    cellular_sig_by_cell_type : dict
        Cell-type -> significant reactions
    rq3_stats_df : pd.DataFrame
        RQ3 statistical tests data (already loaded)
    abundance_dict : dict
        Cell abundance dictionary
    comparison : str
        Comparison name
    
    Returns
    -------
    pd.DataFrame
        Abundance-weighted contribution scores
    """
    logger.info("\n" + "="*80)
    logger.info("PHASE 2: ABUNDANCE-WEIGHTED CONTRIBUTION ANALYSIS")
    logger.info("="*80)
    
    # Filter for this comparison
    flux_df = rq3_stats_df[rq3_stats_df['comparison'] == comparison].copy()
    
    contribution_results = []
    
    for cell_type, sig_rxns in cellular_sig_by_cell_type.items():
        # Get flux data for this cell type
        ct_flux = flux_df[flux_df['cell_type'] == cell_type].copy()
        
        # Calculate metrics
        n_sig = len(sig_rxns)
        n_total = len(ct_flux)
        response_rate = n_sig / n_total if n_total > 0 else 0
        
        # Mean absolute flux change for significant reactions
        sig_flux = ct_flux[ct_flux['reaction_id'].isin(sig_rxns)]
        mean_flux_change = sig_flux['abs_flux_change'].mean() if len(sig_flux) > 0 else 0
        
        # Cell abundance
        abundance = get_cell_abundance(cell_type, abundance_dict)
        
        # Contribution score (weighted by all three factors)
        contribution_score = abundance * response_rate * mean_flux_change
        
        # Alternative: simple weighted response
        simple_weighted = abundance * response_rate * 100  # as percentage
        
        contribution_results.append({
            'cell_type': cell_type,
            'cell_abundance': abundance,
            'n_significant': n_sig,
            'n_total': n_total,
            'response_rate': response_rate,
            'mean_flux_change': mean_flux_change,
            'contribution_score': contribution_score,
            'simple_weighted_response': simple_weighted,
        })
    
    contrib_df = pd.DataFrame(contribution_results)
    contrib_df = contrib_df.sort_values('contribution_score', ascending=False)
    
    # Normalize to percentages
    total_contrib = contrib_df['contribution_score'].sum()
    contrib_df['contribution_percent'] = contrib_df['contribution_score'] / total_contrib * 100
    
    total_simple = contrib_df['simple_weighted_response'].sum()
    contrib_df['simple_weighted_percent'] = contrib_df['simple_weighted_response'] / total_simple * 100
    
    logger.info("\nTop 10 contributors (by abundance-weighted score):")
    logger.info("-" * 80)
    logger.info(f"{'Cell Type':<30s} {'Abundance':>10s} {'Response':>8s} {'Weighted%':>10s}")
    logger.info("-" * 80)
    for idx, row in contrib_df.head(10).iterrows():
        logger.info(f"{row['cell_type']:<30s} {row['cell_abundance']*100:>9.1f}% {row['response_rate']*100:>7.1f}% {row['contribution_percent']:>9.1f}%")
    
    logger.info(f"\nTop 3 contributors account for: {contrib_df.head(3)['contribution_percent'].sum():.1f}% of total")
    
    return contrib_df

# ============================================================================
# Phase 3: Reaction-Level Attribution
# ============================================================================

def classify_reaction_attribution(bulk_sig_reactions, cellular_sig_by_cell_type, 
                                  rq3_stats_df, abundance_dict, comparison='WesternDiet_vs_Chow'):
    """
    Classify each bulk-significant reaction by cellular origin:
      - CELL_TYPE_UNIQUE: Only 1 cell type significant
      - COOPERATIVE: 2-3 cell types significant
      - MULTI_CELLULAR: 4+ cell types significant
      - NON_CELLULAR: No cell types significant (unexpected!)
    
    Parameters
    ----------
    bulk_sig_reactions : pd.DataFrame
        Bulk-significant reactions
    cellular_sig_by_cell_type : dict
        Cell-type -> significant reactions
    rq3_stats_df : pd.DataFrame
        RQ3 statistical tests (already loaded)
    abundance_dict : dict
        Cell abundance dictionary
    comparison : str
        Comparison name
    
    Returns
    -------
    pd.DataFrame
        Attribution classification for each bulk-significant reaction
    """
    logger.info("\n" + "="*80)
    logger.info("PHASE 3: REACTION-LEVEL ATTRIBUTION")
    logger.info("="*80)
    
    # Filter for this comparison
    flux_df = rq3_stats_df[rq3_stats_df['comparison'] == comparison].copy()
    
    attribution_results = []
    
    for idx, rxn_row in bulk_sig_reactions.iterrows():
        rxn_id = rxn_row['ReactionID']
        rxn_name = rxn_row['ReactionName']
        subsystem = rxn_row['Subsystem']
        
        # Find which cell types show this reaction as significant
        cell_types_significant = []
        weighted_contributions = {}
        
        for cell_type, sig_rxns in cellular_sig_by_cell_type.items():
            if rxn_id in sig_rxns:
                cell_types_significant.append(cell_type)
                
                # Get flux change magnitude for this cell type
                ct_flux = flux_df[(flux_df['cell_type'] == cell_type) & 
                                 (flux_df['reaction_id'] == rxn_id)]
                if len(ct_flux) > 0:
                    flux_change = ct_flux.iloc[0]['abs_flux_change']
                    abundance = get_cell_abundance(cell_type, abundance_dict)
                    weighted_contributions[cell_type] = flux_change * abundance
        
        # Classify
        n_cell_types = len(cell_types_significant)
        
        if n_cell_types == 0:
            category = 'NON_CELLULAR'
            primary_driver = 'NONE'
            secondary_drivers = []
        elif n_cell_types == 1:
            category = 'CELL_TYPE_UNIQUE'
            primary_driver = cell_types_significant[0]
            secondary_drivers = []
        elif n_cell_types <= 3:
            category = 'COOPERATIVE'
            # Sort by weighted contribution
            sorted_contributors = sorted(weighted_contributions.items(), 
                                        key=lambda x: x[1], reverse=True)
            primary_driver = sorted_contributors[0][0] if len(sorted_contributors) > 0 else 'UNKNOWN'
            secondary_drivers = [x[0] for x in sorted_contributors[1:]]
        else:
            category = 'MULTI_CELLULAR'
            # Sort by weighted contribution
            sorted_contributors = sorted(weighted_contributions.items(), 
                                        key=lambda x: x[1], reverse=True)
            primary_driver = sorted_contributors[0][0] if len(sorted_contributors) > 0 else 'UNKNOWN'
            secondary_drivers = [x[0] for x in sorted_contributors[1:3]]  # Top 2 secondary
        
        attribution_results.append({
            'reaction_id': rxn_id,
            'reaction_name': rxn_name,
            'subsystem': subsystem,
            'category': category,
            'n_cell_types': n_cell_types,
            'primary_driver': primary_driver,
            'secondary_drivers': '; '.join(secondary_drivers),
            'all_cell_types': '; '.join(cell_types_significant),
        })
    
    attrib_df = pd.DataFrame(attribution_results)
    
    # Summary statistics
    category_counts = attrib_df['category'].value_counts()
    logger.info("\nAttribution Category Summary:")
    logger.info("-" * 80)
    for cat, count in category_counts.items():
        pct = count / len(attrib_df) * 100
        logger.info(f"  {cat:20s}: {count:4d} reactions ({pct:5.1f}%)")
    
    # Primary driver distribution
    logger.info("\nTop 10 Primary Drivers:")
    logger.info("-" * 80)
    primary_counts = attrib_df['primary_driver'].value_counts().head(10)
    for driver, count in primary_counts.items():
        pct = count / len(attrib_df) * 100
        logger.info(f"  {driver:30s}: {count:4d} reactions ({pct:5.1f}%)")
    
    return attrib_df

# ============================================================================
# Phase 4: Pathway-Level Attribution
# ============================================================================

def perform_pathway_attribution(attribution_df, contribution_df):
    """
    Aggregate reaction-level attributions to pathway level.
    
    Parameters
    ----------
    attribution_df : pd.DataFrame
        Reaction-level attribution
    contribution_df : pd.DataFrame
        Cell-type contribution scores
    
    Returns
    -------
    pd.DataFrame
        Pathway-level attribution
    """
    logger.info("\n" + "="*80)
    logger.info("PHASE 4: PATHWAY-LEVEL ATTRIBUTION")
    logger.info("="*80)
    
    # Group by subsystem (pathway)
    pathway_results = []
    
    for subsystem in attribution_df['subsystem'].unique():
        if pd.isna(subsystem) or subsystem == '':
            continue
        
        pathway_rxns = attribution_df[attribution_df['subsystem'] == subsystem]
        
        # Count attributions by category
        category_counts = pathway_rxns['category'].value_counts().to_dict()
        
        # Find primary drivers
        primary_drivers = pathway_rxns['primary_driver'].value_counts()
        
        # Get top 3 drivers
        top_drivers = primary_drivers.head(3).to_dict()
        primary_driver = primary_drivers.index[0] if len(primary_drivers) > 0 else 'UNKNOWN'
        secondary_drivers = list(primary_drivers.index[1:3]) if len(primary_drivers) > 1 else []
        
        # Calculate driver percentages
        total_rxns = len(pathway_rxns)
        driver_percentages = {k: v/total_rxns*100 for k, v in top_drivers.items()}
        
        pathway_results.append({
            'pathway': subsystem,
            'n_reactions': total_rxns,
            'primary_driver': primary_driver,
            'primary_driver_pct': driver_percentages.get(primary_driver, 0),
            'secondary_drivers': '; '.join(secondary_drivers),
            'n_unique': category_counts.get('CELL_TYPE_UNIQUE', 0),
            'n_cooperative': category_counts.get('COOPERATIVE', 0),
            'n_multicellular': category_counts.get('MULTI_CELLULAR', 0),
            'n_non_cellular': category_counts.get('NON_CELLULAR', 0),
        })
    
    pathway_df = pd.DataFrame(pathway_results)
    pathway_df = pathway_df.sort_values('n_reactions', ascending=False)
    
    logger.info("\nTop 15 Pathways by Number of Bulk-Significant Reactions:")
    logger.info("-" * 80)
    logger.info(f"{'Pathway':<50s} {'N_rxns':>7s} {'Primary Driver':>20s}")
    logger.info("-" * 80)
    for idx, row in pathway_df.head(15).iterrows():
        logger.info(f"{row['pathway'][:48]:<50s} {int(row['n_reactions']):>7d} {row['primary_driver']:>20s}")
    
    return pathway_df

# ============================================================================
# Visualization Functions
# ============================================================================

def plot_bulk_cellular_overlap(overlap_df, output_dir):
    """Plot bulk-cellular overlap analysis."""
    logger.info("\nGenerating bulk-cellular overlap visualizations...")
    
    # Sort by overlap percentage
    plot_df = overlap_df.sort_values('overlap_pct', ascending=True).tail(15)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Panel A: Overlap percentage
    ax = axes[0]
    bars = ax.barh(plot_df['cell_type'], plot_df['overlap_pct'], 
                   color='steelblue', alpha=0.7)
    ax.set_xlabel('Overlap with Bulk-Significant Reactions (%)', fontsize=12)
    ax.set_ylabel('Cell Type', fontsize=12)
    ax.set_title('A) Cell Types Recapitulating Bulk Metabolic Changes', 
                fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add abundance as text
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        ax.text(row['overlap_pct'] + 0.5, i, 
               f"{row['cell_abundance']*100:.1f}%", 
               va='center', fontsize=8, color='darkred')
    
    # Panel B: F1 Score vs Abundance (scatter)
    ax = axes[1]
    scatter = ax.scatter(overlap_df['cell_abundance']*100, 
                        overlap_df['f1_score']*100,
                        s=overlap_df['n_overlap']*3,
                        c=overlap_df['overlap_pct'],
                        cmap='viridis',
                        alpha=0.6,
                        edgecolors='black',
                        linewidth=0.5)
    
    # Annotate top cell types
    top_types = overlap_df.nlargest(5, 'f1_score')
    for idx, row in top_types.iterrows():
        ax.annotate(row['cell_type'],
                   (row['cell_abundance']*100, row['f1_score']*100),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.7)
    
    ax.set_xlabel('Cell Abundance (%)', fontsize=12)
    ax.set_ylabel('F1 Score (%)', fontsize=12)
    ax.set_title('B) Detection Performance vs Abundance', 
                fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Overlap %', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase1_bulk_cellular_overlap.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  [OK] Saved: phase1_bulk_cellular_overlap.png")

def plot_contribution_analysis(contribution_df, output_dir):
    """Plot abundance-weighted contribution analysis."""
    logger.info("\nGenerating contribution analysis visualizations...")
    
    # Sort by contribution
    plot_df = contribution_df.sort_values('contribution_percent', ascending=True).tail(15)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Panel A: Stacked bar (abundance × response rate)
    ax = axes[0]
    
    # Create stacked components
    y_pos = np.arange(len(plot_df))
    
    # Calculate components
    abundance_component = plot_df['cell_abundance'] * 100
    response_component = plot_df['response_rate'] * 100
    
    # Plot stacked bars
    p1 = ax.barh(y_pos, abundance_component, color='steelblue', alpha=0.7, 
                label='Cell Abundance')
    p2 = ax.barh(y_pos, response_component, left=abundance_component,
                color='coral', alpha=0.7, label='Response Rate')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df['cell_type'])
    ax.set_xlabel('Percentage', fontsize=12)
    ax.set_title('A) Abundance vs Response Rate', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    
    # Panel B: Contribution percentages
    ax = axes[1]
    bars = ax.barh(plot_df['cell_type'], plot_df['contribution_percent'],
                   color='darkgreen', alpha=0.7)
    ax.set_xlabel('Weighted Contribution to Bulk Variance (%)', fontsize=12)
    ax.set_title('B) Abundance-Weighted Contributions', 
                fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add values on bars
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        ax.text(row['contribution_percent'] + 0.5, i,
               f"{row['contribution_percent']:.1f}%",
               va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase2_contribution_analysis.png'),
               dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  [OK] Saved: phase2_contribution_analysis.png")

def plot_attribution_classification(attribution_df, output_dir):
    """Plot reaction-level attribution classification."""
    logger.info("\nGenerating attribution classification visualizations...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Panel A: Category pie chart
    ax = axes[0]
    category_counts = attribution_df['category'].value_counts()
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    ax.pie(category_counts.values, labels=category_counts.index,
          autopct='%1.1f%%', colors=colors, startangle=90)
    ax.set_title('A) Attribution Categories', 
                fontsize=14, fontweight='bold')
    
    # Panel B: Primary driver bar chart
    ax = axes[1]
    primary_counts = attribution_df['primary_driver'].value_counts().head(15)
    plot_df = primary_counts.sort_values()
    
    bars = ax.barh(range(len(plot_df)), plot_df.values,
                   color='teal', alpha=0.7)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df.index)
    ax.set_xlabel('Number of Reactions', fontsize=12)
    ax.set_title('B) Primary Drivers (Top 15)', 
                fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase3_attribution_classification.png'),
               dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  [OK] Saved: phase3_attribution_classification.png")

def plot_pathway_attribution(pathway_df, output_dir):
    """Plot pathway-level attribution analysis."""
    logger.info("\nGenerating pathway attribution visualizations...")
    
    # Get top 20 pathways
    plot_df = pathway_df.head(20).sort_values('n_reactions', ascending=True)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create stacked horizontal bar chart
    categories = ['n_unique', 'n_cooperative', 'n_multicellular', 'n_non_cellular']
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    labels = ['Unique', 'Cooperative', 'Multi-cellular', 'Unknown'] #Non-cellular
    
    y_pos = np.arange(len(plot_df))
    left = np.zeros(len(plot_df))
    
    for cat, color, label in zip(categories, colors, labels):
        values = plot_df[cat].values
        ax.barh(y_pos, values, left=left, color=color, alpha=0.7, label=label)
        left += values
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df['pathway'])
    ax.set_xlabel('Number of Reactions', fontsize=12)
    ax.set_title('Pathway-Level Attribution (Top 20 Pathways)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase4_pathway_attribution.png'),
               dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  [OK] Saved: phase4_pathway_attribution.png")

# ============================================================================
# Main Integration Pipeline
# ============================================================================

def main():
    """Main integration analysis pipeline."""
    
    parser = argparse.ArgumentParser(
        description='RQ1-RQ2-RQ3 Integration Analysis (REVISED)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
REQUIRED INPUT FILES:
--------------------
1. --rq1_pairwise_stats: RQ1_multidataset_flux_pairwise_stats.csv
   (Bulk-level statistical tests across diets)

2. --rq3_stats: statistical_tests.csv
   (Cell-type-specific statistical tests from RQ3 pipeline)

3. --rq3_aggregation: aggregation_summary.csv
   (Cell counts per cell type from RQ3 pipeline)

DIET MATCHING:
-------------
RQ1 Bulk: WD_vs_SCD (Western Diet vs Standard Chow Diet)
RQ3 Cell: WesternDiet_vs_Chow

Use --bulk_comparison WD_vs_SCD to properly match diets!

CELL ABUNDANCE:
--------------
Default: Calculate from scRNA-seq data (--rq3_aggregation)
Override: Use --use_literature_abundance for literature estimates

Example:
--------
python rq1_rq2_rq3_integration_analysis_REVISED.py \\
  --rq1_pairwise_stats RQ1_multidataset_flux_pairwise_stats.csv \\
  --rq3_stats statistical_tests.csv \\
  --rq3_aggregation aggregation_summary.csv \\
  --bulk_comparison WD_vs_SCD \\
  --cellular_comparison WesternDiet_vs_Chow \\
  --output_dir integration_results
        """
    )
    
    # Input files
    parser.add_argument('--rq1_pairwise_stats', required=True,
                       help='RQ1 pairwise statistical tests CSV')
    parser.add_argument('--rq3_stats', required=True,
                       help='RQ3 statistical tests CSV')
    parser.add_argument('--rq3_aggregation', required=True,
                       help='RQ3 aggregation summary CSV (for cell abundance)')
    
    # Comparisons
    parser.add_argument('--bulk_comparison', default='WD_vs_SCD',
                       choices=['HFD_vs_SCD', 'WD_vs_SCD', 'KD_vs_SCD'],
                       help='Bulk comparison (default: WD_vs_SCD to match WesternDiet)')
    parser.add_argument('--cellular_comparison', default='WesternDiet_vs_Chow',
                       help='Cellular comparison (default: WesternDiet_vs_Chow)')
    
    # Cell abundance
    parser.add_argument('--use_literature_abundance', action='store_true',
                       help='Use literature abundance estimates instead of scRNA-seq derived')
    
    # Thresholds
    parser.add_argument('--fdr_threshold', type=float, default=0.10,
                       help='FDR threshold for bulk significance (default: 0.10)')
    
    # Output
    parser.add_argument('--output_dir', default='integration_analysis',
                       help='Output directory (default: integration_analysis)')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    tables_dir = os.path.join(args.output_dir, 'tables')
    figures_dir = os.path.join(args.output_dir, 'figures')
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    logger.info("="*80)
    logger.info("RQ1-RQ2-RQ3 INTEGRATION ANALYSIS (REVISED)")
    logger.info("="*80)
    logger.info(f"\nInput Files:")
    logger.info(f"  RQ1 pairwise stats: {args.rq1_pairwise_stats}")
    logger.info(f"  RQ3 statistics:     {args.rq3_stats}")
    logger.info(f"  RQ3 aggregation:    {args.rq3_aggregation}")
    logger.info(f"\nComparisons:")
    logger.info(f"  Bulk:     {args.bulk_comparison}")
    logger.info(f"  Cellular: {args.cellular_comparison}")
    logger.info(f"\nParameters:")
    logger.info(f"  FDR threshold: {args.fdr_threshold}")
    logger.info(f"  Cell abundance: {'Literature' if args.use_literature_abundance else 'scRNA-seq derived'}")
    logger.info(f"  Output directory: {args.output_dir}")
    
    # ========================================================================
    # Calculate Cell Abundance
    # ========================================================================
    
    if args.use_literature_abundance:
        logger.info("\nUsing LITERATURE cell abundance estimates")
        abundance_dict = LITERATURE_CELL_ABUNDANCE
    else:
        logger.info("\nCalculating cell abundance from scRNA-seq data")
        abundance_dict = calculate_cell_abundance_from_scrna(args.rq3_aggregation)
    
    # Save abundance to file
    abundance_df = pd.DataFrame([
        {'cell_type': ct, 'abundance': ab, 'abundance_percent': ab*100}
        for ct, ab in sorted(abundance_dict.items(), key=lambda x: x[1], reverse=True)
    ])
    abundance_df.to_csv(os.path.join(tables_dir, 'cell_abundance_used.csv'), index=False)
    logger.info(f"\n  [OK] Saved cell abundance to: {tables_dir}/cell_abundance_used.csv")
    
    # ========================================================================
    # Load Data
    # ========================================================================
    
    # Load bulk-significant reactions (RQ1)
    bulk_sig = load_rq1_bulk_significant_reactions(
        args.rq1_pairwise_stats,
        comparison=args.bulk_comparison,
        fdr_threshold=args.fdr_threshold
    )
    
    # Load cell-type-specific significant reactions (RQ3)
    rq3_stats, cellular_sig = load_rq3_cellular_significant_reactions(
        args.rq3_stats,
        comparison=args.cellular_comparison
    )
    
    # ========================================================================
    # Phase 1: Bulk-Cellular Overlap
    # ========================================================================
    
    overlap_df = calculate_bulk_cellular_overlap(bulk_sig, cellular_sig, abundance_dict)
    overlap_df.to_csv(os.path.join(tables_dir, 'phase1_bulk_cellular_overlap.csv'), 
                     index=False)
    plot_bulk_cellular_overlap(overlap_df, figures_dir)
    
    # ========================================================================
    # Phase 2: Abundance-Weighted Contributions
    # ========================================================================
    
    contribution_df = calculate_abundance_weighted_contributions(
        cellular_sig, rq3_stats, abundance_dict, args.cellular_comparison
    )
    contribution_df.to_csv(os.path.join(tables_dir, 'phase2_contribution_analysis.csv'),
                          index=False)
    plot_contribution_analysis(contribution_df, figures_dir)
    
    # ========================================================================
    # Phase 3: Reaction-Level Attribution
    # ========================================================================
    
    attribution_df = classify_reaction_attribution(
        bulk_sig, cellular_sig, rq3_stats, abundance_dict, args.cellular_comparison
    )
    attribution_df.to_csv(os.path.join(tables_dir, 'phase3_reaction_attribution.csv'),
                         index=False)
    plot_attribution_classification(attribution_df, figures_dir)
    
    # ========================================================================
    # Phase 4: Pathway-Level Attribution
    # ========================================================================
    
    pathway_df = perform_pathway_attribution(attribution_df, contribution_df)
    pathway_df.to_csv(os.path.join(tables_dir, 'phase4_pathway_attribution.csv'),
                     index=False)
    plot_pathway_attribution(pathway_df, figures_dir)
    
    # ========================================================================
    # Summary Report
    # ========================================================================
    
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info(f"\nOutput directory: {args.output_dir}")
    logger.info(f"  Tables: {tables_dir}")
    logger.info(f"  Figures: {figures_dir}")
    logger.info("\nGenerated files:")
    logger.info("  Tables:")
    logger.info("    - cell_abundance_used.csv")
    logger.info("    - phase1_bulk_cellular_overlap.csv")
    logger.info("    - phase2_contribution_analysis.csv")
    logger.info("    - phase3_reaction_attribution.csv")
    logger.info("    - phase4_pathway_attribution.csv")
    logger.info("  Figures:")
    logger.info("    - phase1_bulk_cellular_overlap.png")
    logger.info("    - phase2_contribution_analysis.png")
    logger.info("    - phase3_attribution_classification.png")
    logger.info("    - phase4_pathway_attribution.png")
    
    logger.info("\n" + "="*80)
    logger.info("KEY FINDINGS")
    logger.info("="*80)
    
    # Top contributors
    top3_contrib = contribution_df.head(3)
    logger.info("\nTop 3 Cell Type Contributors:")
    for idx, row in top3_contrib.iterrows():
        logger.info(f"  {idx+1}. {row['cell_type']:30s}: {row['contribution_percent']:5.1f}% "
                   f"(abundance: {row['cell_abundance']*100:.1f}%, response: {row['response_rate']*100:.1f}%)")
    
    # Attribution summary
    logger.info("\nReaction Attribution Summary:")
    cat_counts = attribution_df['category'].value_counts()
    for cat, count in cat_counts.items():
        pct = count / len(attribution_df) * 100
        logger.info(f"  {cat:20s}: {count:4d} reactions ({pct:5.1f}%)")
    
    logger.info("\n" + "="*80)
    
if __name__ == '__main__':
    main()
