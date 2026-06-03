#!/usr/bin/env python3
"""
BULK-TO-SINGLE-CELL VALIDATION ANALYSIS - LOCAL VERSION
========================================================

This script validates that cell-type-specific metabolic flux changes recapitulate
bulk tissue-level responses.

REQUIREMENTS:
- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scipy

INSTALL DEPENDENCIES:
pip install pandas numpy matplotlib seaborn scipy

USAGE:
1. Edit the "CONFIGURATION" section below with your file paths
2. Run: python bulk_validation_LOCAL.py

Author: PhD Dissertation Analysis
Date: March 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os
import sys

# =============================================================================
# CONFIGURATION - EDIT THESE PATHS FOR YOUR SYSTEM
# =============================================================================

# Input files
BULK_FLUX_FILE = "Processing_outputs/Step_1_RQ1/batch_corrected/flux_analysis/reaction_flux_comparison_extended_batch_corrected.csv" 
SC_STATS_FILE = "Processing_outputs/Step_3_RQ3/statistics/statistical_tests.csv" 
ABUNDANCE_FILE = "Processing_outputs/Step_3_RQ3/Step_3b_results_rq1_rq3_integration/tables/cell_abundance_used.csv"


# Output directory (will be created if doesn't exist)
OUTPUT_DIR = "Processing_outputs/Step_3_RQ3/Hierarchical_Analysis"

# Comparison to analyze
BULK_COMPARISON = "WD_vs_SCD"  # Options: "WD_vs_SCD", "HFD_vs_SCD", "KD_vs_SCD"
SC_COMPARISON = "WesternDiet_vs_Chow"  # Corresponding single-cell comparison

# Plotting parameters
FIGURE_DPI = 300
FIGURE_SIZE = (8, 8)

# =============================================================================
# DO NOT EDIT BELOW THIS LINE (unless you know what you're doing)
# =============================================================================

# Set publication-ready plotting style
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['figure.dpi'] = FIGURE_DPI


def check_files_exist():
    """Check that all required input files exist."""
    print("Checking input files...")
    
    missing_files = []
    for filepath in [BULK_FLUX_FILE, SC_STATS_FILE, ABUNDANCE_FILE]:
        if not os.path.exists(filepath):
            missing_files.append(filepath)
            print(f"  ✗ NOT FOUND: {filepath}")
        else:
            print(f"  ✓ Found: {filepath}")
    
    if missing_files:
        print("\n❌ ERROR: Missing required files!")
        print("\nPlease update the file paths in the CONFIGURATION section at the top of this script.")
        print("\nMissing files:")
        for f in missing_files:
            print(f"  - {f}")
        sys.exit(1)
    
    print("\n✓ All input files found!\n")


def load_bulk_data(bulk_file, comparison):
    """
    Load bulk tissue flux data from extended comparison file.
    
    Parameters:
    -----------
    bulk_file : str
        Path to RQ1_multidataset_reaction_flux_comparison_extended.csv
    comparison : str
        Which comparison to use (e.g., "WD_vs_SCD", "HFD_vs_SCD")
    
    Returns:
    --------
    DataFrame with columns: reaction_id, flux_change_bulk
    """
    print("Loading bulk tissue flux data...")
    print(f"  File: {bulk_file}")
    print(f"  Comparison: {comparison}")
    
    df = pd.read_csv(bulk_file)
    print(f"  Loaded {len(df)} reactions")
    
    # Map comparison name to column name
    diff_col = f"Diff({comparison.replace('_vs_', '-')})"
    
    if diff_col not in df.columns:
        print(f"\n❌ ERROR: Column '{diff_col}' not found in bulk file!")
        print(f"\nAvailable difference columns:")
        for col in df.columns:
            if col.startswith('Diff('):
                print(f"  - {col}")
        sys.exit(1)
    
    # Extract reaction ID and flux change
    bulk_df = df[['ReactionID', diff_col]].copy()
    bulk_df.columns = ['reaction_id', 'flux_change_bulk']
    
    # Remove NaN values
    bulk_df = bulk_df.dropna()
    
    print(f"  Valid reactions: {len(bulk_df)}")
    print(f"  Mean flux change: {bulk_df['flux_change_bulk'].mean():.4f}")
    
    return bulk_df


def load_and_aggregate_singlecell(sc_stats_file, abundance_file, comparison):
    """
    Load single-cell flux changes and aggregate weighted by cell abundance.
    
    Parameters:
    -----------
    sc_stats_file : str
        Path to RQ3_statistical_tests.csv
    abundance_file : str
        Path to RQ3_cell_abundance_used.csv
    comparison : str
        Which comparison to use (e.g., "WesternDiet_vs_Chow")
    
    Returns:
    --------
    DataFrame with columns: reaction_id, flux_change_aggregated
    """
    print("\nLoading and aggregating single-cell flux data...")
    print(f"  Stats file: {sc_stats_file}")
    print(f"  Abundance file: {abundance_file}")
    print(f"  Comparison: {comparison}")
    
    # Load statistical tests
    df_stats = pd.read_csv(sc_stats_file)
    
    # Filter for specified comparison
    df_stats = df_stats[df_stats['comparison'] == comparison].copy()
    print(f"  Loaded {len(df_stats)} reaction-cell type pairs")
    
    # Load cell abundances
    df_abund = pd.read_csv(abundance_file)
    abund_dict = dict(zip(df_abund['cell_type'], df_abund['abundance']))
    print(f"  Loaded {len(abund_dict)} cell type abundances")
    
    # Get unique reactions
    reactions = df_stats['reaction_id'].unique()
    print(f"  Unique reactions: {len(reactions)}")
    
    # Aggregate flux changes weighted by abundance
    print("  Aggregating flux changes...")
    aggregated_data = []
    
    for rxn in reactions:
        # Get all cell types for this reaction
        rxn_data = df_stats[df_stats['reaction_id'] == rxn]
        
        # Weighted average of flux changes
        total_weight = 0
        weighted_change = 0
        
        for _, row in rxn_data.iterrows():
            cell_type = row['cell_type']
            if cell_type in abund_dict:
                weight = abund_dict[cell_type]
                weighted_change += row['flux_change'] * weight
                total_weight += weight
        
        aggregated_data.append({
            'reaction_id': rxn,
            'flux_change_aggregated': weighted_change,
            'total_weight': total_weight
        })
    
    df_agg = pd.DataFrame(aggregated_data)
    
    print(f"  Aggregated {len(df_agg)} reactions")
    print(f"  Mean total weight: {df_agg['total_weight'].mean():.3f}")
    print(f"  Mean flux change: {df_agg['flux_change_aggregated'].mean():.4f}")
    
    return df_agg[['reaction_id', 'flux_change_aggregated']]


def calculate_validation_metrics(bulk_df, agg_df):
    """
    Calculate validation metrics between bulk and aggregated single-cell.
    
    Parameters:
    -----------
    bulk_df : DataFrame
        Bulk flux changes
    agg_df : DataFrame
        Aggregated single-cell flux changes
    
    Returns:
    --------
    dict with validation metrics and merged data
    """
    print("\nCalculating validation metrics...")
    
    # Merge datasets
    df = bulk_df.merge(agg_df, on='reaction_id', how='inner')
    print(f"  Common reactions: {len(df)}")
    
    if len(df) < 2:
        print("\n❌ ERROR: Not enough common reactions for correlation!")
        sys.exit(1)
    
    # Remove NaN values
    mask = ~(np.isnan(df['flux_change_bulk']) | np.isnan(df['flux_change_aggregated']))
    df = df[mask]
    print(f"  Valid reactions for correlation: {len(df)}")
    
    # 1. Pearson correlation
    r, p = pearsonr(df['flux_change_bulk'], df['flux_change_aggregated'])
    print(f"\n  Pearson r = {r:.3f} (p = {p:.2e})")
    
    # 2. Directional agreement
    df['same_direction'] = np.sign(df['flux_change_bulk']) == np.sign(df['flux_change_aggregated'])
    df['both_nonzero'] = (df['flux_change_bulk'] != 0) & (df['flux_change_aggregated'] != 0)
    
    directional_agreement = df[df['both_nonzero']]['same_direction'].mean() * 100
    print(f"  Directional agreement: {directional_agreement:.1f}%")
    
    # 3. RMSE
    rmse = np.sqrt(np.mean((df['flux_change_bulk'] - df['flux_change_aggregated'])**2))
    print(f"  RMSE: {rmse:.4f}")
    
    # 4. Mean absolute error
    mae = np.mean(np.abs(df['flux_change_bulk'] - df['flux_change_aggregated']))
    print(f"  MAE: {mae:.4f}")
    
    results = {
        'pearson_r': r,
        'pearson_p': p,
        'directional_agreement': directional_agreement,
        'rmse': rmse,
        'mae': mae,
        'n_reactions': len(df),
        'data': df
    }
    
    return results


def create_validation_plot(results, output_dir):
    """Create validation scatter plot."""
    print("\nGenerating validation plot...")
    
    df = results['data']
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    # Plot all reactions
    ax.scatter(df['flux_change_bulk'], 
              df['flux_change_aggregated'],
              alpha=0.4, s=30, color='#4E79A7', edgecolors='white', linewidth=0.5)
    
    # Add identity line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0, linewidth=1.5, label='Perfect agreement')
    
    # Add regression line
    z = np.polyfit(df['flux_change_bulk'], df['flux_change_aggregated'], 1)
    p = np.poly1d(z)
    ax.plot(lims, p(lims), 'r-', alpha=0.5, linewidth=1.5, label=f'Best fit (r={results["pearson_r"]:.3f})')
    
    # Add metrics text
    metrics_text = (
        f"Pearson r = {results['pearson_r']:.3f}\n"
        f"p = {results['pearson_p']:.2e}\n"
        f"Directional agreement = {results['directional_agreement']:.1f}%\n"
        f"RMSE = {results['rmse']:.4f}\n"
        f"n = {results['n_reactions']} reactions"
    )
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))
    
    ax.set_xlabel('Bulk Flux Change (WD - SCD)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Aggregated Single-Cell\nFlux Change (WD - SCD)', fontsize=12, fontweight='bold')
    ax.set_title('Bulk-to-Single-Cell Validation', fontsize=14, fontweight='bold', pad=15)
    ax.legend(frameon=True, fancybox=False, edgecolor='black', fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'bulk_singlecell_validation.png')
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    
    plt.close()


def save_results(results, output_dir):
    """Save validation results and data."""
    print("\nSaving results...")
    
    # Save validation data
    data_file = os.path.join(output_dir, 'bulk_singlecell_validation_data.csv')
    results['data'].to_csv(data_file, index=False)
    print(f"  Saved data: {data_file}")
    
    # Save metrics
    metrics_file = os.path.join(output_dir, 'bulk_singlecell_validation_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BULK-TO-SINGLE-CELL VALIDATION METRICS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Comparison: {BULK_COMPARISON} vs {SC_COMPARISON}\n\n")
        f.write(f"Pearson correlation:        r = {results['pearson_r']:.4f}\n")
        f.write(f"P-value:                    p = {results['pearson_p']:.2e}\n")
        f.write(f"Directional agreement:      {results['directional_agreement']:.1f}%\n")
        f.write(f"RMSE:                       {results['rmse']:.4f}\n")
        f.write(f"MAE:                        {results['mae']:.4f}\n")
        f.write(f"\n")
        f.write(f"Total reactions:            {results['n_reactions']}\n")
        f.write("\n")
        f.write("="*80 + "\n")
        f.write("INTERPRETATION\n")
        f.write("="*80 + "\n\n")
        
        if results['pearson_r'] > 0.7:
            f.write("Strong positive correlation indicates that single-cell metabolic\n")
            f.write("modeling successfully captures bulk tissue-level responses.\n")
        elif results['pearson_r'] > 0.3:
            f.write("Moderate positive correlation suggests that single-cell modeling\n")
            f.write("partially captures bulk tissue behavior, but tissue-level integration\n")
            f.write("and emergent properties also play important roles.\n")
        elif results['pearson_r'] > -0.3:
            f.write("Weak correlation suggests that simple abundance-weighted aggregation\n")
            f.write("does not fully capture bulk tissue behavior. This indicates:\n")
            f.write("- Intercellular metabolite exchange is important\n")
            f.write("- Tissue-level organization creates emergent properties\n")
            f.write("- Cell-cell interactions affect metabolic responses\n")
        else:
            f.write("Negative correlation indicates systematic differences between\n")
            f.write("bulk measurements and aggregated single-cell predictions.\n")
            f.write("This suggests tissue-level integration effects that cannot be\n")
            f.write("captured by simple weighted aggregation of single-cell responses.\n")
    
    print(f"  Saved metrics: {metrics_file}")


def main():
    """Main analysis workflow."""
    print("="*80)
    print("BULK-TO-SINGLE-CELL VALIDATION ANALYSIS")
    print("="*80)
    print()
    
    # Check files exist
    check_files_exist()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    # Run analysis
    bulk_df = load_bulk_data(BULK_FLUX_FILE, BULK_COMPARISON)
    agg_df = load_and_aggregate_singlecell(SC_STATS_FILE, ABUNDANCE_FILE, SC_COMPARISON)
    results = calculate_validation_metrics(bulk_df, agg_df)
    
    # Generate outputs
    create_validation_plot(results, OUTPUT_DIR)
    save_results(results, OUTPUT_DIR)
    
    # Final summary
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nKey Findings:")
    print(f"  Pearson r = {results['pearson_r']:.3f} (p < 0.001)" if results['pearson_p'] < 0.001 else f"  Pearson r = {results['pearson_r']:.3f} (p = {results['pearson_p']:.3f})")
    print(f"  Directional agreement = {results['directional_agreement']:.1f}%")
    print(f"  RMSE = {results['rmse']:.4f}")
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print(f"  - bulk_singlecell_validation.png       (validation plot)")
    print(f"  - bulk_singlecell_validation_data.csv  (full comparison data)")
    print(f"  - bulk_singlecell_validation_metrics.txt (summary statistics)")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
