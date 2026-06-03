#!/usr/bin/env python3
"""
HIERARCHICAL ATTRIBUTION ANALYSIS - LOCAL VERSION
==================================================

This script performs hierarchical attribution of metabolic responses to Western Diet,
classifying cell types across three biological hierarchies: Function, Location, Lineage.

REQUIREMENTS:
- Python 3.8+
- pandas
- numpy

INSTALL DEPENDENCIES:
pip install pandas numpy

USAGE:
1. Edit the "CONFIGURATION" section below with your file path
2. Run: python hierarchical_attribution_LOCAL.py

Author: PhD Dissertation Analysis
Date: March 2026
"""

import pandas as pd
import numpy as np
import json
import os
import sys

# =============================================================================
# CONFIGURATION - EDIT THESE PATHS FOR YOUR SYSTEM
# =============================================================================

# Input file (use absolute path or path relative to this script)
#CONTRIBUTION_FILE = "RQ3_phase2_contribution_analysis.csv"
CONTRIBUTION_FILE = "Processing_outputs/Step_3_RQ3/Step_3b_results_rq1_rq3_integration/tables/phase2_contribution_analysis.csv"

# Output directory (will be created if doesn't exist)
OUTPUT_DIR = "Processing_outputs/Step_3_RQ3/Hierarchical_Analysis"

# Which contribution metric to use
# Options: "contribution_percent", "simple_weighted_percent", "contribution_score"
CONTRIBUTION_METRIC = "contribution_percent"

# =============================================================================
# HIERARCHICAL CELL TYPE GROUPINGS
# =============================================================================

# Function Hierarchy: What do the cells do?
FUNCTION_HIERARCHY = {
    'Immune': [
        'Timd4+ resKC',
        'Cd207+, Trem2+ Mo-KC',
        'Cd207-, Trem2+ Mo-KC',
        'Cx3cr1+, Ccr2+ MdM',
        'Trem2+, Spp1+ MdM',
        'Ly6c2+, Ccr2+ Mo',
        'Ly6c2-, Spn+ Mo',
        'Transitioning Mo',
        'T cells',
        'B cells',
        'Neutrophiles',
        'Dendrites',
        'DCs',
        'Macrophages',
        'MCs',
        'Mast cells',
        'Cycling'
    ],
    'Metabolic': [
        'Hepatocytes',
        'Cholangiocytes'
    ],
    'Structural': [
        'LECs',
        'qHSCs',
        'aHSCs',
        'cAMP qHSCs'
    ]
}

# Location Hierarchy: Where are the cells?
LOCATION_HIERARCHY = {
    'Sinusoidal': [
        'LECs',
        'Timd4+ resKC',
        'Cd207+, Trem2+ Mo-KC',
        'Cd207-, Trem2+ Mo-KC',
        'qHSCs',
        'aHSCs',
        'cAMP qHSCs'
    ],
    'Parenchymal': [
        'Hepatocytes'
    ],
    'Portal': [
        'Cholangiocytes'
    ],
    'Circulating': [
        'Cx3cr1+, Ccr2+ MdM',
        'Trem2+, Spp1+ MdM',
        'Ly6c2+, Ccr2+ Mo',
        'Ly6c2-, Spn+ Mo',
        'Transitioning Mo',
        'T cells',
        'B cells',
        'Neutrophiles',
        'Dendrites',
        'DCs',
        'Macrophages',
        'MCs',
        'Mast cells'
    ]
}

# Lineage Hierarchy: What is their developmental origin?
LINEAGE_HIERARCHY = {
    'Myeloid': [
        'Timd4+ resKC',
        'Cd207+, Trem2+ Mo-KC',
        'Cd207-, Trem2+ Mo-KC',
        'Cx3cr1+, Ccr2+ MdM',
        'Trem2+, Spp1+ MdM',
        'Ly6c2+, Ccr2+ Mo',
        'Ly6c2-, Spn+ Mo',
        'Transitioning Mo',
        'Neutrophiles',
        'Dendrites',
        'DCs',
        'Macrophages',
        'MCs'
    ],
    'Lymphoid': [
        'T cells',
        'B cells'
    ],
    'Epithelial': [
        'Hepatocytes',
        'Cholangiocytes'
    ],
    'Endothelial': [
        'LECs'
    ],
    'Mesenchymal': [
        'qHSCs',
        'aHSCs',
        'cAMP qHSCs',
        'Mast cells'
    ]
}

# =============================================================================
# DO NOT EDIT BELOW THIS LINE (unless you know what you're doing)
# =============================================================================


def check_files_exist():
    """Check that required input file exists."""
    print("Checking input files...")
    
    if not os.path.exists(CONTRIBUTION_FILE):
        print(f"  ✗ NOT FOUND: {CONTRIBUTION_FILE}")
        print("\n❌ ERROR: Missing required file!")
        print("\nPlease update the file path in the CONFIGURATION section at the top of this script.")
        sys.exit(1)
    else:
        print(f"  ✓ Found: {CONTRIBUTION_FILE}")
    
    print()


def load_contribution_data(contrib_file, metric):
    """Load cell type contribution data."""
    print("Loading cell type contribution data...")
    print(f"  File: {contrib_file}")
    
    df = pd.read_csv(contrib_file)
    print(f"  Loaded {len(df)} cell types")
    
    # Check if metric column exists
    if metric not in df.columns:
        print(f"\n❌ ERROR: Column '{metric}' not found in contribution file!")
        print(f"\nAvailable columns:")
        for col in df.columns:
            print(f"  - {col}")
        print(f"\nPlease update CONTRIBUTION_METRIC in the CONFIGURATION section.")
        sys.exit(1)
    
    print(f"  Using metric: {metric}")
    print(f"  Total contribution: {df[metric].sum():.2f}%")
    
    return df


def assign_to_hierarchy(contrib_df, hierarchy_dict, hierarchy_name, metric):
    """Assign each cell type to hierarchical groups and calculate contributions."""
    print(f"\nAssigning to {hierarchy_name} hierarchy...")
    
    # Set cell_type as index for easier lookup
    contrib_indexed = contrib_df.set_index('cell_type')
    
    # Calculate contribution for each group
    group_contributions = {}
    assignments = {}
    
    for group_name, group_cells in hierarchy_dict.items():
        total_contribution = 0
        cells_found = []
        
        for cell in group_cells:
            if cell in contrib_indexed.index:
                total_contribution += contrib_indexed.loc[cell, metric]
                assignments[cell] = group_name
                cells_found.append(cell)
        
        group_contributions[group_name] = total_contribution
        print(f"  {group_name:20s}: {total_contribution:6.1f}% ({len(cells_found)} cell types)")
    
    # Calculate percentages (normalize to 100%)
    total = sum(group_contributions.values())
    if total > 0:
        percentages = {k: (v/total*100) for k, v in group_contributions.items()}
    else:
        percentages = {k: 0 for k in group_contributions.keys()}
    
    # Find dominant group
    if percentages:
        dominant = max(percentages.items(), key=lambda x: x[1])
        print(f"\n  → DOMINANT: {dominant[0]} ({dominant[1]:.1f}%)")
    
    return assignments, percentages, dominant


def generate_hierarchical_summary(contrib_df, metric):
    """Generate hierarchical attribution summary across all three hierarchies."""
    
    results = {}
    
    # Function hierarchy
    func_assign, func_pct, func_dominant = assign_to_hierarchy(
        contrib_df, FUNCTION_HIERARCHY, "FUNCTION", metric
    )
    results['function'] = {
        'assignments': func_assign,
        'percentages': func_pct,
        'dominant': func_dominant
    }
    
    # Location hierarchy
    loc_assign, loc_pct, loc_dominant = assign_to_hierarchy(
        contrib_df, LOCATION_HIERARCHY, "LOCATION", metric
    )
    results['location'] = {
        'assignments': loc_assign,
        'percentages': loc_pct,
        'dominant': loc_dominant
    }
    
    # Lineage hierarchy
    lin_assign, lin_pct, lin_dominant = assign_to_hierarchy(
        contrib_df, LINEAGE_HIERARCHY, "LINEAGE", metric
    )
    results['lineage'] = {
        'assignments': lin_assign,
        'percentages': lin_pct,
        'dominant': lin_dominant
    }
    
    return results


def print_summary(results):
    """Print hierarchical attribution summary."""
    print("\n" + "="*80)
    print("HIERARCHICAL ATTRIBUTION SUMMARY")
    print("="*80)
    
    print("\nFUNCTION Hierarchy:")
    print("-" * 40)
    for group, pct in sorted(results['function']['percentages'].items(), 
                             key=lambda x: x[1], reverse=True):
        print(f"  {group:20s} {pct:6.1f}%")
    print(f"\n  → {results['function']['dominant'][0]}-Dominant: {results['function']['dominant'][1]:.1f}%")
    
    print("\n\nLOCATION Hierarchy:")
    print("-" * 40)
    for group, pct in sorted(results['location']['percentages'].items(), 
                             key=lambda x: x[1], reverse=True):
        print(f"  {group:20s} {pct:6.1f}%")
    print(f"\n  → {results['location']['dominant'][0]}-Dominant: {results['location']['dominant'][1]:.1f}%")
    
    print("\n\nLINEAGE Hierarchy:")
    print("-" * 40)
    for group, pct in sorted(results['lineage']['percentages'].items(), 
                             key=lambda x: x[1], reverse=True):
        print(f"  {group:20s} {pct:6.1f}%")
    print(f"\n  → {results['lineage']['dominant'][0]}-Led: {results['lineage']['dominant'][1]:.1f}%")


def save_results(results, output_dir):
    """Save hierarchical attribution results."""
    print("\n\nSaving results...")
    
    # Save as CSV for each hierarchy
    for hierarchy in ['function', 'location', 'lineage']:
        df = pd.DataFrame([
            {'Group': group, 'Contribution_Percent': pct}
            for group, pct in results[hierarchy]['percentages'].items()
        ]).sort_values('Contribution_Percent', ascending=False)
        
        output_file = os.path.join(output_dir, f'{hierarchy}_attribution.csv')
        df.to_csv(output_file, index=False)
        print(f"  Saved: {output_file}")
    
    # Save summary text file
    summary_file = os.path.join(output_dir, 'hierarchical_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("HIERARCHICAL ATTRIBUTION SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write("THE THREE DOMINANT PATTERNS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"1. FUNCTION:  {results['function']['dominant'][0]}-Dominant "
                f"({results['function']['dominant'][1]:.1f}%)\n")
        f.write(f"2. LOCATION:  {results['location']['dominant'][0]}-Dominant "
                f"({results['location']['dominant'][1]:.1f}%)\n")
        f.write(f"3. LINEAGE:   {results['lineage']['dominant'][0]}-Led "
                f"({results['lineage']['dominant'][1]:.1f}%)\n")
        
        f.write("\n\n" + "="*80 + "\n")
        f.write("DETAILED BREAKDOWN\n")
        f.write("="*80 + "\n\n")
        
        for hierarchy in ['function', 'location', 'lineage']:
            f.write(f"\n{hierarchy.upper()} Hierarchy:\n")
            f.write("-" * 40 + "\n")
            for group, pct in sorted(results[hierarchy]['percentages'].items(),
                                    key=lambda x: x[1], reverse=True):
                f.write(f"  {group:20s} {pct:6.1f}%\n")
    
    print(f"  Saved: {summary_file}")
    
    # Save as JSON
    json_file = os.path.join(output_dir, 'hierarchical_results.json')
    json_data = {
        hierarchy: {
            'percentages': results[hierarchy]['percentages'],
            'dominant_group': results[hierarchy]['dominant'][0],
            'dominant_percent': results[hierarchy]['dominant'][1]
        }
        for hierarchy in ['function', 'location', 'lineage']
    }
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"  Saved: {json_file}")


def main():
    """Main analysis workflow."""
    print("="*80)
    print("HIERARCHICAL ATTRIBUTION ANALYSIS")
    print("="*80)
    print()
    
    # Check files exist
    check_files_exist()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    # Load data
    contrib_df = load_contribution_data(CONTRIBUTION_FILE, CONTRIBUTION_METRIC)
    
    # Generate hierarchical attribution
    results = generate_hierarchical_summary(contrib_df, CONTRIBUTION_METRIC)
    
    # Print and save results
    print_summary(results)
    save_results(results, OUTPUT_DIR)
    
    # Final summary
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("\nKey Finding - THE THREE DOMINANT PATTERNS:")
    print(f"  1. {results['function']['dominant'][0]}-Dominant: {results['function']['dominant'][1]:.1f}%")
    print(f"  2. {results['location']['dominant'][0]}-Dominant: {results['location']['dominant'][1]:.1f}%")
    print(f"  3. {results['lineage']['dominant'][0]}-Led: {results['lineage']['dominant'][1]:.1f}%")
    print("\nGenerated files:")
    print(f"  - function_attribution.csv      (Function hierarchy percentages)")
    print(f"  - location_attribution.csv      (Location hierarchy percentages)")
    print(f"  - lineage_attribution.csv       (Lineage hierarchy percentages)")
    print(f"  - hierarchical_summary.txt      (Text summary)")
    print(f"  - hierarchical_results.json     (JSON format for figures script)")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
