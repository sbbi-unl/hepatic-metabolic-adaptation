#!/usr/bin/env python3
"""
RQ4: Master Pipeline Orchestrator
==================================

"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List
from datetime import datetime

import pandas as pd

# ============================================================================
# CONDITION NAME MAPPING
# ============================================================================
# Microbiome data uses: ND_SCD, DD_HFD
# Hepatic expression file uses: SCD_C57BL6J_GSM..., HFD_C57BL6J_GSM...
# This mapping ensures correct column access in Stage 2

HEPATIC_CONDITION_MAP = {
    "ND_SCD": "SCD_C57BL6J_GSM5534348_m_lim_j16800",
    "DD_HFD": "HFD_C57BL6J_GSM5534354_m_lim_j17543"
}

def get_hepatic_condition(microbiome_condition):
    """Map microbiome condition name to hepatic expression condition."""
    return HEPATIC_CONDITION_MAP.get(microbiome_condition, microbiome_condition)




################################################################################
# CONFIGURATION
################################################################################

DEFAULT_CONFIG = {
    "project_name": "RQ4_Host_Microbiome_Integration",
    "data": {
        "metatranscriptome": "Meta_GSE104913.csv",
        "agora_models_dir": "/path/to/AGORA2",
        "hepatic_model": "iMM1415.json",
        "hepatic_expression": "GSE182668_expression.csv",
        "diet_bounds": "expanded_diet_bounds_flat.json",
        "gene_mapping": "mouse_entrez_to_symbol.csv"
    },
    "analysis_params": {
        "expression_cols": ["ND_SCD", "DD_HFD"],
        "abundance_method": "total_expression",
        "micom_tradeoff": 0.5,
        "eflux_quantile": 0.95,
        "eflux_floor": 0.1,
        "eflux_cap": 1000.0,
        "flux_threshold": 0.01
    },
    "output": {
        "base_dir": "results_rq4_complete",
        "subdirs": {
            "community": "01_community_modeling",
            "hepatic": "02_hepatic_integration",
            "attribution": "03_attribution_analysis",
            "figures": "04_publication_figures",
            "reports": "05_analysis_reports"
        }
    }
}


################################################################################
# UTILITY FUNCTIONS
################################################################################

def create_directory_structure(base_dir: str, subdirs: Dict[str, str]):
    """
    Create organized output directory structure.

    """
    print("[INFO] Creating directory structure...")
    
    os.makedirs(base_dir, exist_ok=True)
    
    dir_map = {}
    for key, subdir_name in subdirs.items():
        full_path = os.path.join(base_dir, subdir_name)
        os.makedirs(full_path, exist_ok=True)
        dir_map[key] = full_path
        print(f"  ✓ {full_path}")
    
    return dir_map


def run_subprocess(cmd: List[str], description: str) -> bool:
    """
    Execute subprocess with error handling and logging.

    """
    print(f"\n{'='*80}")
    print(f"[STAGE] {description}")
    print(f"{'='*80}")
    print(f"[CMD] {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        
        if result.stderr:
            print("[STDERR]")
            print(result.stderr)
        
        print(f"\n[SUCCESS] {description} completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {description} failed!")
        print(f"[STDOUT] {e.stdout}")
        print(f"[STDERR] {e.stderr}")
        return False
    
    except Exception as e:
        print(f"\n[ERROR] Unexpected error in {description}: {e}")
        return False


################################################################################
# PIPELINE STAGES
################################################################################

def stage_1_community_modeling(
    config: Dict,
    output_dirs: Dict[str, str]
) -> bool:
    """
    Stage 1: Microbiome community modeling with MICOM.

    """
    print("\n" + "="*80)
    print("STAGE 1: MICROBIOME COMMUNITY MODELING")
    print("="*80)
    
    cmd = [
        "python", "scripts/rq4_microbiome_community_modeling_v2026.py", #rq4_microbiome_community_modeling.py
        "--metatranscriptome", config["data"]["metatranscriptome"],
        "--agora_dir", config["data"]["agora_models_dir"],
        "--expression_cols"] + config["analysis_params"]["expression_cols"] + [
        "--abundance_method", config["analysis_params"]["abundance_method"],
        "--tradeoff", str(config["analysis_params"]["micom_tradeoff"]),
        "--results_dir", output_dirs["community"]
    ]
    
    # Add optional medium file if available
    if config["data"].get("diet_bounds"):
        cmd.extend(["--medium_json", config["data"]["diet_bounds"]])
    
    
    # Add optional metabolite filtering control
    # Default is True (filter), so only add flag if explicitly set to False
    if not config["analysis_params"].get("filter_metabolites", True):
        cmd.append("--no-filter_metabolites")
    
    return run_subprocess(cmd, "Microbiome Community Modeling")


def stage_2_hepatic_integration(
    config: Dict,
    output_dirs: Dict[str, str],
    portal_metabolites_file: str
) -> bool:
    """
    Stage 2: Integrate microbiome outputs with hepatic model.
    Runs for each dietary condition (with and without microbiome).
    Uses the biologically corrected V7 integration script.
    """
    print("\n" + "="*80)
    print("STAGE 2: HEPATIC MODEL INTEGRATION (V13 CORRECTED)")
    print("="*80)
    
    success_all = True
    
    for condition in config["analysis_params"]["expression_cols"]:
        cmd = [
            "python", "scripts/rq4_hepatic_integration_CORRECTED_v13.py",
            "--hepatic_model", config["data"]["hepatic_model"],
            "--expression_data", config["data"]["hepatic_expression"],
            "--portal_metabolites", portal_metabolites_file,
            "--condition", condition,
            "--diet_bounds", config["data"]["diet_bounds"],   # <--- THE MISSING LINK IS RESTORED
            "--eflux_floor", str(config["analysis_params"].get("eflux_floor", 0.1)),
            "--eflux_cap", str(config["analysis_params"].get("eflux_cap", 1000.0)),
            "--flux_threshold", str(config["analysis_params"].get("flux_threshold", 0.01)),
            "--objective_mode", str(config["analysis_params"].get("objective_mode", "atpm")),
            "--portal_scaling", str(config["analysis_params"].get("portal_scaling", 0.1)),
            "--portal_mode", str(config["analysis_params"].get("portal_mode", "forced")),
            "--results_dir", os.path.join(output_dirs["hepatic"], f"condition_{condition}")
        ]
        
        # Handle the pFBA flag (V7 uses pFBA by default, so we pass --no-pfba if set to false)
        if not config["analysis_params"].get("use_pfba", True):
            cmd.append("--no-pfba")
        
        success = run_subprocess(cmd, f"Hepatic Integration - {condition}")
        success_all = success_all and success
    
    return success_all


def stage_3_attribution_analysis(
    config: Dict,
    output_dirs: Dict[str, str]
) -> bool:
    """
    Stage 3: Differential flux attribution analysis.
    
    Quantifies relative contributions of genetics, diet, and microbiome.
    """
    print("\n" + "="*80)
    print("STAGE 3: DIFFERENTIAL FLUX ATTRIBUTION")
    print("="*80)
    
    # Determine baseline and treatment conditions
    conditions = config["analysis_params"]["expression_cols"]
    
    if len(conditions) < 2:
        print("[WARNING] Need at least 2 conditions for attribution analysis")
        return False
    
    baseline = conditions[0]  # Typically SCD
    treatment = conditions[1]  # Typically HFD
    
    cmd = [
        "python", "scripts/rq4_attribution_analysis.py",
        "--hepatic_results_dir", output_dirs["hepatic"],
        "--baseline_condition", baseline,
        "--treatment_condition", treatment,
        "--results_dir", output_dirs["attribution"]
    ]
    
    return run_subprocess(cmd, "Attribution Analysis")


def stage_3b_pathway_enrichment(
    config: Dict,
    output_dirs: Dict[str, str]
) -> bool:
    """
    Stage 3b: Pathway enrichment analysis.

    """
    print("\n" + "="*80)
    print("STAGE 3B: PATHWAY ENRICHMENT ANALYSIS")
    print("="*80)
    
    # Define paths
    attribution_csv = os.path.join(
        output_dirs["attribution"],
        "flux_attribution_analysis.csv"
    )
    
    pathway_output = os.path.join(
        output_dirs["attribution"],
        "pathway_enrichment"
    )
    
    # Check prerequisites
    if not os.path.exists(attribution_csv):
        print(f"[ERROR] Attribution file not found: {attribution_csv}")
        print("[ERROR] Stage 3 must complete successfully before pathway enrichment")
        return False
    
    if not os.path.exists(config["data"]["hepatic_model"]):
        print(f"[ERROR] Model file not found: {config['data']['hepatic_model']}")
        return False
    
    # Build command
    cmd = [
        "python", "scripts/rq4_pathway_enrichment_module.py",
        "--model", config["data"]["hepatic_model"],
        "--attribution", attribution_csv,
        "--output", pathway_output
    ]
    
    success = run_subprocess(cmd, "Pathway Enrichment Analysis")
    
    # Verify outputs if successful
    if success:
        expected_files = [
            'pathway_enrichment_results.csv',
            'compartment_enrichment_results.csv',
            'pathway_synergy_analysis.csv',
            'PATHWAY_ANALYSIS_SUMMARY.txt'
        ]
        
        print("\n[INFO] Verifying critical pathway enrichment outputs...")
        missing = []
        for filename in expected_files:
            filepath = os.path.join(pathway_output, filename)
            if os.path.exists(filepath):
                print(f"  ✓ {filename}")
            else:
                print(f"  ✗ {filename} (missing)")
                missing.append(filename)
        
        if missing:
            print(f"\n[WARNING] {len(missing)} output file(s) missing")
            return False
    
    return success


def stage_4_integrated_reporting(
    config: Dict,
    output_dirs: Dict[str, str]
) -> bool:
    """
    Stage 4: Generate integrated report and publication figures.

    """
    print("\n" + "="*80)
    print("STAGE 4: INTEGRATED REPORTING")
    print("="*80)
    
    report_file = os.path.join(output_dirs["reports"], "RQ4_Analysis_Report.txt")
    
    try:
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("RQ4: HOST-MICROBIOME METABOLIC INTEGRATION\n")
            f.write("Complete Analysis Report\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("ANALYSIS CONFIGURATION\n")
            f.write("-"*80 + "\n")
            f.write(json.dumps(config, indent=2))
            f.write("\n\n")
            
            # Section 1: Community Modeling Results
            f.write("SECTION 1: MICROBIOME COMMUNITY MODELING\n")
            f.write("-"*80 + "\n")
            
            abundance_file = os.path.join(output_dirs["community"], "species_abundances.csv")
            if os.path.exists(abundance_file):
                abundance_df = pd.read_csv(abundance_file)
                f.write(f"Species analyzed: {len(abundance_df)}\n")
                f.write("\nTop 10 most abundant species (averaged across conditions):\n")
                
                abundance_cols = [c for c in abundance_df.columns if '_abundance' in c]
                if abundance_cols:
                    abundance_df['mean_abundance'] = abundance_df[abundance_cols].mean(axis=1)
                    top10 = abundance_df.nlargest(10, 'mean_abundance')
                    for idx, row in top10.iterrows():
                        f.write(f"  {row['species']:50s} {row['mean_abundance']:.4f}\n")
            
            f.write("\n\n")
            
            # Section 2: Portal Metabolites
            f.write("SECTION 2: PORTAL METABOLITE PRODUCTION\n")
            f.write("-"*80 + "\n")
            
            portal_file = os.path.join(output_dirs["community"], "portal_metabolite_production.csv")
            if os.path.exists(portal_file):
                portal_df = pd.read_csv(portal_file)
                f.write(f"Portal metabolites detected: {portal_df['Metabolite'].nunique()}\n")
                f.write(f"Conditions analyzed: {portal_df['Condition'].nunique()}\n\n")
                
                f.write("Summary by condition:\n")
                for condition in portal_df['Condition'].unique():
                    cond_data = portal_df[portal_df['Condition'] == condition]
                    f.write(f"\n{condition}:\n")
                    for idx, row in cond_data.iterrows():
                        f.write(f"  {row['Metabolite']:20s} {row['Flux']:10.4f} mmol/gDW/h\n")
            
            f.write("\n\n")
            
            # Section 3: Hepatic Integration
            f.write("SECTION 3: HEPATIC MODEL INTEGRATION\n")
            f.write("-"*80 + "\n")
            
            for condition in config["analysis_params"]["expression_cols"]:
                cond_dir = os.path.join(output_dirs["hepatic"], f"condition_{condition}")
                flux_file = os.path.join(cond_dir, f"{condition}_flux_comparison.csv")
                
                if os.path.exists(flux_file):
                    flux_df = pd.read_csv(flux_file)
                    
                    f.write(f"\n{condition}:\n")
                    f.write(f"  Total reactions analyzed: {len(flux_df)}\n")
                    
                    if 'microbiome_attributable' in flux_df.columns:
                        n_affected = flux_df['microbiome_attributable'].sum()
                        pct_affected = (n_affected / len(flux_df)) * 100
                        f.write(f"  Reactions affected by microbiome: {n_affected} ({pct_affected:.1f}%)\n")
            
            f.write("\n\n")
            
            # Section 4: Attribution Analysis
            f.write("SECTION 4: FLUX ATTRIBUTION ANALYSIS\n")
            f.write("-"*80 + "\n")
            
            attribution_file = os.path.join(output_dirs["attribution"], "flux_attribution_analysis.csv")
            if os.path.exists(attribution_file):
                attr_df = pd.read_csv(attribution_file)
                
                significant = attr_df[attr_df['delta_total'].abs() > config["analysis_params"]["flux_threshold"]]
                
                f.write(f"Reactions with significant changes: {len(significant)}/{len(attr_df)}\n\n")
                
                f.write("Dominant driver distribution:\n")
                driver_counts = significant['dominant_driver'].value_counts()
                for driver, count in driver_counts.items():
                    pct = (count / len(significant)) * 100
                    f.write(f"  {driver:20s}: {count:4d} ({pct:5.1f}%)\n")
                
                #f.write(f"\nMean microbiome contribution: {significant['pct_microbiome_contribution'].mean():.1f}%\n")
                #f.write(f"Median microbiome contribution: {significant['pct_microbiome_contribution'].median():.1f}%\n")
                #pct_microbiome_contribution_legacy
                f.write(f"\nMean microbiome contribution: {significant['pct_microbiome_contribution_legacy'].mean():.1f}%\n")
                f.write(f"Median microbiome contribution: {significant['pct_microbiome_contribution_legacy'].median():.1f}%\n")
            
            f.write("\n\n")
            
            # Section 4b: Pathway Enrichment Analysis
            f.write("SECTION 4B: PATHWAY ENRICHMENT ANALYSIS\n")
            f.write("-"*80 + "\n")
            
            pathway_enrichment_file = os.path.join(
                output_dirs["attribution"],
                "pathway_enrichment",
                "pathway_enrichment_results.csv"
            )
            
            if os.path.exists(pathway_enrichment_file):
                pathway_df = pd.read_csv(pathway_enrichment_file)
                
                # Significant pathways
                sig_pathways = pathway_df[pathway_df['significant'] == True]
                f.write(f"Pathways analyzed: {len(pathway_df)}\n")
                f.write(f"Significantly enriched pathways: {len(sig_pathways)}\n\n")
                
                if len(sig_pathways) > 0:
                    f.write("Top 10 enriched pathways:\n")
                    for idx, row in sig_pathways.head(10).iterrows():
                        f.write(f"  {row['subsystem'][:50]:50s} ")
                        f.write(f"{row['n_significant']:3d}/{row['n_reactions']:3d} ")
                        f.write(f"(enrichment={row['enrichment_ratio']:.2f}x, ")
                        f.write(f"p={row['p_adjusted']:.2e})\n")
                
                # Synergy statistics
                f.write("\nPathway synergy summary:\n")
                if 'synergistic_pct' in pathway_df.columns:
                    highly_syn = len(pathway_df[pathway_df['synergistic_pct'] > 70])
                    f.write(f"  Highly synergistic pathways (>70%): {highly_syn}\n")
                    
                if 'antagonistic_pct' in pathway_df.columns:
                    highly_ant = len(pathway_df[pathway_df['antagonistic_pct'] > 70])
                    f.write(f"  Highly antagonistic pathways (>70%): {highly_ant}\n")
            else:
                f.write("[Note: Pathway enrichment analysis not run or failed]\n")
            
            # Compartment enrichment
            compartment_file = os.path.join(
                output_dirs["attribution"],
                "pathway_enrichment",
                "compartment_enrichment_results.csv"
            )
            
            if os.path.exists(compartment_file):
                comp_df = pd.read_csv(compartment_file)
                sig_comp = comp_df[comp_df['significant'] == True]
                
                f.write(f"\nCompartment enrichment:\n")
                if len(sig_comp) > 0:
                    for idx, row in sig_comp.iterrows():
                        f.write(f"  {row['compartment']:15s} ")
                        f.write(f"{row['n_significant']:3d}/{row['n_reactions']:3d} ")
                        f.write(f"(MB variance={row['mean_microbiome_variance']:.1f}%, ")
                        f.write(f"p={row['p_adjusted']:.2e})\n")
                else:
                    f.write("  No compartments significantly enriched (FDR < 0.05)\n")
            
            f.write("\n\n")
            
            # Section 5: Key Findings
            f.write("SECTION 5: KEY FINDINGS & BIOLOGICAL INTERPRETATION\n")
            f.write("-"*80 + "\n")
            f.write("[Note: Add your biological interpretation here based on results]\n\n")
            
            f.write("="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"\n[SAVED] Integrated analysis report: {report_file}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to generate report: {e}")
        return False


################################################################################
# MAIN PIPELINE
################################################################################

def main():
    parser = argparse.ArgumentParser(
        description="RQ4 Master Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Complete RQ4 Workflow:
---------------------
This master script runs the entire RQ4 analysis pipeline:

Stage 1: Microbiome community modeling (MICOM)
Stage 2: Hepatic model integration
Stage 3: Differential flux attribution
Stage 3b: Pathway enrichment analysis (automatic after Stage 3)
Stage 4: Integrated reporting & visualization

Examples:
--------
# Run with default configuration (all stages including pathway enrichment)
python rq4_master_pipeline.py --config rq4_config.json

# Run specific stages only
python rq4_master_pipeline.py --config rq4_config.json --stages 1 2

# Skip completed stages
python rq4_master_pipeline.py --config rq4_config.json --skip-stage 1

# Skip pathway enrichment
python rq4_master_pipeline.py --config rq4_config.json --skip-pathway-enrichment

Note: Stage 3b (Pathway Enrichment) runs automatically after Stage 3 completes
      successfully. It adds ~30 seconds to pipeline runtime.
"""
    )
    
    parser.add_argument(
        '--config',
        required=True,
        help='Configuration JSON file'
    )
    parser.add_argument(
        '--stages',
        type=int,
        nargs='+',
        default=None,
        help='Specific stages to run (1-4). If not specified, runs all.'
    )
    parser.add_argument(
        '--skip-stage',
        type=int,
        action='append',
        default=[],
        help='Stages to skip (can specify multiple times)'
    )
    parser.add_argument(
        '--skip-pathway-enrichment',
        action='store_true',
        help='Skip pathway enrichment analysis (Stage 3b)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print("[INFO] Loading configuration...")
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    print(f"[INFO] Project: {config.get('project_name', 'RQ4 Analysis')}")
    
    # Create directory structure
    output_dirs = create_directory_structure(
        config["output"]["base_dir"],
        config["output"]["subdirs"]
    )
    
    # Determine which stages to run
    all_stages = [1, 2, 3, 4]
    
    if args.stages:
        stages_to_run = [s for s in args.stages if s not in args.skip_stage]
    else:
        stages_to_run = [s for s in all_stages if s not in args.skip_stage]
    
    print(f"\n[INFO] Stages to run: {stages_to_run}")
    
    # Track success
    results = {}
    
    # Execute pipeline
    print("\n" + "="*80)
    print(f"STARTING RQ4 ANALYSIS PIPELINE")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Stage 1
    if 1 in stages_to_run:
        results[1] = stage_1_community_modeling(config, output_dirs)
        if not results[1]:
            print("\n[ERROR] Stage 1 failed. Aborting pipeline.")
            sys.exit(1)
    
    # Stage 2
    if 2 in stages_to_run:
        portal_file = os.path.join(
            output_dirs["community"],
            "portal_metabolites_for_hepatic_model.json"
        )
        
        if not os.path.exists(portal_file):
            print(f"\n[ERROR] Portal metabolites file not found: {portal_file}")
            print("[ERROR] Run Stage 1 first or provide file manually.")
            sys.exit(1)
        
        results[2] = stage_2_hepatic_integration(config, output_dirs, portal_file)
        if not results[2]:
            print("\n[WARNING] Stage 2 failed. Continuing with remaining stages.")
    
    # Stage 3
    if 3 in stages_to_run:
        results[3] = stage_3_attribution_analysis(config, output_dirs)
        if not results[3]:
            print("\n[WARNING] Stage 3 failed. Continuing with remaining stages.")
    
    # Stage 3b: Pathway Enrichment (runs automatically after successful Stage 3)
    if 3 in stages_to_run and results.get(3, False) and not args.skip_pathway_enrichment:
        results['3b'] = stage_3b_pathway_enrichment(config, output_dirs)
        if not results['3b']:
            print("\n[WARNING] Stage 3b (Pathway Enrichment) failed. Continuing with remaining stages.")
    elif 3 in stages_to_run and args.skip_pathway_enrichment:
        print("\n[INFO] Skipping Stage 3b (Pathway Enrichment) - User requested skip")
        results['3b'] = None  # None indicates skipped
    elif 3 in stages_to_run and not results.get(3, False):
        print("\n[INFO] Skipping Stage 3b (Pathway Enrichment) - Stage 3 did not complete successfully")
        results['3b'] = False
    
    # Stage 4
    if 4 in stages_to_run:
        results[4] = stage_4_integrated_reporting(config, output_dirs)
    
    # Final summary
    print("\n" + "="*80)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*80)
    
    for stage in sorted(results.keys(), key=lambda x: (0 if x == '3b' else int(x), str(x))):
        success = results[stage]
        
        if success is None:
            status = "⊘ SKIPPED"
        elif success:
            status = "✓ SUCCESS"
        else:
            status = "✗ FAILED"
        
        stage_name = f"Stage {stage}"
        if stage == '3b':
            stage_name = "Stage 3b (Pathway Enrichment)"
        
        print(f"{stage_name}: {status}")
    
    print("\n" + "="*80)
    print("RQ4 PIPELINE COMPLETE")
    print("="*80)
    print(f"\nResults directory: {config['output']['base_dir']}")
    print("\nNext steps:")
    print("1. Review analysis reports in 05_analysis_reports/")
    print("2. Examine figures in 04_publication_figures/")
    if results.get('3b') == True:
        print("3. Review pathway enrichment results in 03_attribution_analysis/pathway_enrichment/")
        print("   - pathway_enrichment_results.csv (enriched pathways)")
        print("   - compartment_enrichment_results.csv (compartment analysis)")
        print("   - pathway_synergy_analysis.csv (synergy classification)")
        print("   - 4 publication-ready figures (.png)")
        print("   - PATHWAY_ANALYSIS_SUMMARY.txt (publication summary)")
        print("4. Interpret biological findings in context of RQ1-3")
        print("5. Prepare manuscript sections")
    else:
        print("3. Interpret biological findings in context of RQ1-3")
        print("4. Prepare manuscript sections")


if __name__ == "__main__":
    main()
