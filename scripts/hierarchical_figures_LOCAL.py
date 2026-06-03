#!/usr/bin/env python3
"""
HIERARCHICAL ATTRIBUTION FIGURES - LOCAL VERSION
=================================================

This script generates publication-ready figures using actual hierarchical attribution data.

REQUIREMENTS:
- Python 3.8+
- matplotlib
- numpy
- pandas
- (optional) seaborn for enhanced styling

INSTALL DEPENDENCIES:
pip install matplotlib numpy pandas seaborn

USAGE:
1. First run hierarchical_attribution_LOCAL.py to generate hierarchical_results.json
2. Edit the "CONFIGURATION" section below with your paths
3. Run: python hierarchical_figures_LOCAL.py

Author: PhD Dissertation Analysis
Date: March 2026
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
import sys

# Try to import seaborn for better styling (optional)
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Note: seaborn not installed - using default matplotlib styling")

# =============================================================================
# CONFIGURATION - EDIT THESE PATHS FOR YOUR SYSTEM
# =============================================================================

# Input file (JSON results from hierarchical_attribution_LOCAL.py)
RESULTS_FILE = "Processing_outputs/Step_3_RQ3/Hierarchical_Analysis/hierarchical_results.json"

# Output directory (will be created if doesn't exist)
OUTPUT_DIR = "Processing_outputs/Step_3_RQ3/Hierarchical_Analysis"

# Figure settings
FIGURE_DPI = 300
MAIN_FIGURE_SIZE = (18, 6)      # For 3-panel pie chart figure
SUMMARY_FIGURE_SIZE = (10, 6)   # For summary bar chart
COMPREHENSIVE_SIZE = (18, 12)   # For 9-panel comprehensive figure

# Color schemes (professional, colorblind-friendly)
FUNC_COLORS = {
    'Structural': '#E15759',   # Red - dominant
    'Immune': '#4E79A7',       # Blue
    'Metabolic': '#F28E2B',    # Orange
}

LOC_COLORS = {
    'Sinusoidal': '#E15759',    # Red - dominant
    'Circulating': '#4E79A7',    # Blue
    'Portal': '#F28E2B',        # Orange
    'Parenchymal': '#76B7B2',   # Teal
}

LIN_COLORS = {
    'Endothelial': '#E15759',   # Red - dominant
    'Myeloid': '#4E79A7',       # Blue
    'Mesenchymal': '#F28E2B',   # Orange
    'Epithelial': '#76B7B2',    # Teal
    'Lymphoid': '#59A14F',      # Green
}

# =============================================================================
# DO NOT EDIT BELOW THIS LINE (unless you know what you're doing)
# =============================================================================

# Set publication-ready plotting style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5


def check_files_exist():
    """Check that required input file exists."""
    print("Checking input files...")
    
    if not os.path.exists(RESULTS_FILE):
        print(f"  ✗ NOT FOUND: {RESULTS_FILE}")
        print("\n❌ ERROR: Missing hierarchical results file!")
        print("\nYou need to run hierarchical_attribution_LOCAL.py first to generate:")
        print(f"  {RESULTS_FILE}")
        print("\nOr update the RESULTS_FILE path in the CONFIGURATION section.")
        sys.exit(1)
    else:
        print(f"  ✓ Found: {RESULTS_FILE}")
    
    print()


def load_results(results_file):
    """Load hierarchical attribution results from JSON."""
    print(f"Loading results from: {results_file}")
    
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Validate structure
        required_keys = ['function', 'location', 'lineage']
        for key in required_keys:
            if key not in results:
                print(f"\n❌ ERROR: Missing '{key}' in results file!")
                sys.exit(1)
        
        print("  ✓ Results loaded successfully")
        print(f"\n  Function dominant: {results['function']['dominant_group']} ({results['function']['dominant_percent']:.1f}%)")
        print(f"  Location dominant: {results['location']['dominant_group']} ({results['location']['dominant_percent']:.1f}%)")
        print(f"  Lineage dominant:  {results['lineage']['dominant_group']} ({results['lineage']['dominant_percent']:.1f}%)")
        
        return results
        
    except json.JSONDecodeError as e:
        print(f"\n❌ ERROR: Invalid JSON file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: Could not load results: {e}")
        sys.exit(1)


def create_hierarchical_piecharts(results, output_dir):
    """
    Create three pie charts showing dominant groups in each hierarchy.
    This is the main text Figure 1.
    """
    print("\nCreating hierarchical pie charts (Main Text Figure)...")
    
    fig, axes = plt.subplots(1, 3, figsize=MAIN_FIGURE_SIZE)
    fig.suptitle('Hierarchical Attribution of Metabolic Responses to Western Diet',
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Function hierarchy
    func_data = results['function']['percentages']
    func_labels = [f"{k}\n{v:.1f}%" for k, v in func_data.items()]
    func_colors = [FUNC_COLORS.get(k, '#B07AA1') for k in func_data.keys()]
    
    wedges1, texts1, autotexts1 = axes[0].pie(
        func_data.values(),
        labels=func_labels,
        colors=func_colors,
        autopct='',
        startangle=90,
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )
    axes[0].set_title('Function Hierarchy', fontsize=14, fontweight='bold', pad=20)
    
    # Location hierarchy
    loc_data = results['location']['percentages']
    loc_labels = [f"{k}\n{v:.1f}%" for k, v in loc_data.items()]
    loc_colors = [LOC_COLORS.get(k, '#B07AA1') for k in loc_data.keys()]
    
    wedges2, texts2, autotexts2 = axes[1].pie(
        loc_data.values(),
        labels=loc_labels,
        colors=loc_colors,
        autopct='',
        startangle=90,
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )
    axes[1].set_title('Location Hierarchy', fontsize=14, fontweight='bold', pad=20)
    
    # Lineage hierarchy
    lin_data = results['lineage']['percentages']
    lin_labels = [f"{k}\n{v:.1f}%" for k, v in lin_data.items()]
    lin_colors = [LIN_COLORS.get(k, '#B07AA1') for k in lin_data.keys()]
    
    wedges3, texts3, autotexts3 = axes[2].pie(
        lin_data.values(),
        labels=lin_labels,
        colors=lin_colors,
        autopct='',
        startangle=90,
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )
    axes[2].set_title('Lineage Hierarchy', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'hierarchical_pie_charts.png')
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    
    plt.close()


def create_summary_figure(results, output_dir):
    """Create summary bar chart with the three dominant patterns."""
    print("Creating summary bar chart...")
    
    fig, ax = plt.subplots(figsize=SUMMARY_FIGURE_SIZE)
    
    # Prepare data
    categories = ['Function', 'Location', 'Lineage']
    dominant_groups = [
        results['function']['dominant_group'],
        results['location']['dominant_group'],
        results['lineage']['dominant_group']
    ]
    percentages = [
        results['function']['dominant_percent'],
        results['location']['dominant_percent'],
        results['lineage']['dominant_percent']
    ]
    
    # Create bars
    bars = ax.barh(categories, percentages, color=['#E15759', '#4E79A7', '#F28E2B'], height=0.6)
    
    # Add value labels on bars
    for i, (bar, pct, group) in enumerate(zip(bars, percentages, dominant_groups)):
        ax.text(pct + 2, i, f'{group}: {pct:.1f}%',
                va='center', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Contribution Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Dominant Hierarchical Patterns in Metabolic Responses to Western Diet',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'hierarchical_summary_bars.png')
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    
    plt.close()


def create_comprehensive_figure(results, output_dir):
    """Create 3x3 panel comprehensive figure for supplementary material."""
    print("Creating comprehensive 9-panel figure (Supplementary)...")
    
    fig = plt.figure(figsize=COMPREHENSIVE_SIZE)
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
    
    # Row 1: Function hierarchy
    # Panel A: Bar chart
    ax1 = fig.add_subplot(gs[0, 0])
    func_data = pd.DataFrame([
        {'Group': k, 'Percentage': v}
        for k, v in results['function']['percentages'].items()
    ]).sort_values('Percentage', ascending=True)
    
    bars1 = ax1.barh(func_data['Group'], func_data['Percentage'],
                     color=[FUNC_COLORS.get(g, '#B07AA1') for g in func_data['Group']])
    ax1.set_xlabel('Contribution (%)', fontsize=10)
    ax1.set_title('A. Function Hierarchy', fontsize=12, fontweight='bold', loc='left')
    ax1.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax1.set_axisbelow(True)
    
    # Panel B: Pie chart
    ax2 = fig.add_subplot(gs[0, 1])
    colors1 = [FUNC_COLORS.get(k, '#B07AA1') for k in results['function']['percentages'].keys()]
    wedges, texts, autotexts = ax2.pie(
        results['function']['percentages'].values(),
        labels=list(results['function']['percentages'].keys()),
        colors=colors1,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 9}
    )
    ax2.set_title('B. Function Distribution', fontsize=12, fontweight='bold', loc='left')
    
    # Panel C: Dominant pattern
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    ax3.text(0.5, 0.5,
             f"DOMINANT\n\n{results['function']['dominant_group']}\n\n{results['function']['dominant_percent']:.1f}%",
             ha='center', va='center', fontsize=14, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='#E15759', alpha=0.3, pad=1.5, edgecolor='black', linewidth=2))
    
    # Row 2: Location hierarchy
    # Panel D: Bar chart
    ax4 = fig.add_subplot(gs[1, 0])
    loc_data = pd.DataFrame([
        {'Group': k, 'Percentage': v}
        for k, v in results['location']['percentages'].items()
    ]).sort_values('Percentage', ascending=True)
    
    bars2 = ax4.barh(loc_data['Group'], loc_data['Percentage'],
                     color=[LOC_COLORS.get(g, '#B07AA1') for g in loc_data['Group']])
    ax4.set_xlabel('Contribution (%)', fontsize=10)
    ax4.set_title('C. Location Hierarchy', fontsize=12, fontweight='bold', loc='left')
    ax4.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax4.set_axisbelow(True)
    
    # Panel E: Pie chart
    ax5 = fig.add_subplot(gs[1, 1])
    colors2 = [LOC_COLORS.get(k, '#B07AA1') for k in results['location']['percentages'].keys()]
    wedges, texts, autotexts = ax5.pie(
        results['location']['percentages'].values(),
        labels=list(results['location']['percentages'].keys()),
        colors=colors2,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 9}
    )
    ax5.set_title('D. Location Distribution', fontsize=12, fontweight='bold', loc='left')
    
    # Panel F: Dominant pattern
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    ax6.text(0.5, 0.5,
             f"DOMINANT\n\n{results['location']['dominant_group']}\n\n{results['location']['dominant_percent']:.1f}%",
             ha='center', va='center', fontsize=14, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='#E15759', alpha=0.3, pad=1.5, edgecolor='black', linewidth=2))
    
    # Row 3: Lineage hierarchy
    # Panel G: Bar chart
    ax7 = fig.add_subplot(gs[2, 0])
    lin_data = pd.DataFrame([
        {'Group': k, 'Percentage': v}
        for k, v in results['lineage']['percentages'].items()
    ]).sort_values('Percentage', ascending=True)
    
    bars3 = ax7.barh(lin_data['Group'], lin_data['Percentage'],
                     color=[LIN_COLORS.get(g, '#B07AA1') for g in lin_data['Group']])
    ax7.set_xlabel('Contribution (%)', fontsize=10)
    ax7.set_title('E. Lineage Hierarchy', fontsize=12, fontweight='bold', loc='left')
    ax7.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax7.set_axisbelow(True)
    
    # Panel H: Pie chart
    ax8 = fig.add_subplot(gs[2, 1])
    colors3 = [LIN_COLORS.get(k, '#B07AA1') for k in results['lineage']['percentages'].keys()]
    wedges, texts, autotexts = ax8.pie(
        results['lineage']['percentages'].values(),
        labels=list(results['lineage']['percentages'].keys()),
        colors=colors3,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 9}
    )
    ax8.set_title('F. Lineage Distribution', fontsize=12, fontweight='bold', loc='left')
    
    # Panel I: Dominant pattern
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    ax9.text(0.5, 0.5,
             f"DOMINANT\n\n{results['lineage']['dominant_group']}\n\n{results['lineage']['dominant_percent']:.1f}%",
             ha='center', va='center', fontsize=14, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='#E15759', alpha=0.3, pad=1.5, edgecolor='black', linewidth=2))
    
    fig.suptitle('Comprehensive Hierarchical Attribution Analysis',
                 fontsize=16, fontweight='bold', y=0.995)
    
    output_file = os.path.join(output_dir, 'hierarchical_comprehensive.png')
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    
    plt.close()


def main():
    """Main figure generation workflow."""
    print("="*80)
    print("HIERARCHICAL FIGURES GENERATION")
    print("="*80)
    print()
    
    # Check files exist
    check_files_exist()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    # Load results
    results = load_results(RESULTS_FILE)
    
    # Create figures
    print()
    create_hierarchical_piecharts(results, OUTPUT_DIR)
    create_summary_figure(results, OUTPUT_DIR)
    create_comprehensive_figure(results, OUTPUT_DIR)
    
    # Final summary
    print("\n" + "="*80)
    print("✓ FIGURE GENERATION COMPLETE")
    print("="*80)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("\nGenerated figures:")
    print("  1. hierarchical_pie_charts.png         - Main text figure (3 pie charts)")
    print("  2. hierarchical_summary_bars.png       - Summary bar chart")
    print("  3. hierarchical_comprehensive.png      - 9-panel comprehensive figure")
    print("\nAll figures generated at 300 DPI for publication quality.")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
