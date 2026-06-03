#!/usr/bin/env python3
"""
===============================================================================
MASTER RQ INTEGRATION PIPELINE
===============================================================================

Pipeline Order:
----------------------------------------------
1. RQ3: Single-cell E-Flux analysis (CORE GENERATOR)
   - Creates cell-type-specific metabolic models and raw data
   - Generates: aggregation_summary.csv, statistical_tests.csv
   - Independent - runs first as the foundation

2. RQ1-RQ2-RQ3 Full Integration (INTERPRETER)
   - Explains WHY bulk liver changes (RQ1) occur based on cellular data (RQ3)
   - Links bulk-level metabolic rewiring to cell-type-specific responses
   - Depends on: RQ1 pairwise stats + RQ3 outputs

3. RQ2-RQ3 Integration: Multi-strain analysis (GENERALIZER)
   - Tests if cellular attribution findings hold across genetic diversity
   - Validates whether cellular architecture is conserved or strain-specific
   - Depends on: RQ1 stats + RQ2 per-strain flux files + RQ3 outputs

===============================================================================
"""

import os
import sys
import json
import argparse
import subprocess
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import shutil

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default strain list for RQ2 analysis
DEFAULT_STRAINS = [
    'C57BL6J', '129S1SvImJ', 'AJ', 'CASTEiJ', 'DBA2J',
    'NODShiLtJ', 'NZOHlLtJ', 'PWKPhJ', 'WSBEiJ'
]

# Default parameters
DEFAULT_CONFIG = {
    # RQ3 Parameters
    'rq3': {
        'eflux_quantile': 0.99,
        'eflux_floor': 0.001,
        'eflux_cap': 10000,
        'normalization_strategy': 'per_condition',
        'min_cells': 20,
        'baseline_condition': 'Chow',
        'test_conditions': 'WesternDiet,HighFat,Ketogenic',
        'fold_change_threshold': 1.5,
        'abs_change_threshold': 0.1,
        'run_fva': False,
        'fva_fraction': 0.9,
        'objective': 'BIOMASS_mm_1_no_glygln',
    },
    # RQ2-RQ3 Integration Parameters
    'rq2_rq3': {
        'contribution_threshold': 0.15,
        'fdr_threshold': 0.10,
        'bulk_comparison': 'WD_vs_SCD',
        'cellular_comparison': 'WesternDiet_vs_Chow',
    },
    # RQ1-RQ2-RQ3 Integration Parameters
    'rq1_rq2_rq3': {
        'bulk_comparison': 'WD_vs_SCD',
        'cellular_comparison': 'WesternDiet_vs_Chow',
        'fdr_threshold': 0.10,
        'use_literature_abundance': False,
    }
}

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(output_dir: str, log_level: str = 'INFO') -> logging.Logger:
    """Configure logging with file and console handlers."""
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'pipeline_{timestamp}.log')
    
    # Create logger
    logger = logging.getLogger('MasterPipeline')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # File handler (detailed) - use UTF-8 encoding
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(fh_format)
    
    # Console handler (concise) - handle Windows encoding issues
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, log_level.upper()))
    ch_format = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(ch_format)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    logger.info(f"Logging initialized: {log_file}")
    return logger


# =============================================================================
# INPUT VALIDATION
# =============================================================================

def validate_file_exists(filepath: str, description: str, logger: logging.Logger) -> bool:
    """Check if a required file exists."""
    if os.path.isfile(filepath):
        logger.info(f"  [OK] Found: {description}")
        logger.debug(f"    Path: {filepath}")
        return True
    else:
        logger.error(f"  [X] Missing: {description}")
        logger.error(f"    Expected: {filepath}")
        return False


def validate_rq3_inputs(config: Dict, logger: logging.Logger) -> bool:
    """Validate inputs for RQ3 analysis."""
    logger.info("\nValidating RQ3 inputs...")
    
    required_files = [
        (config['sc_data'], "Single-cell expression matrix"),
        (config['sc_metadata'], "Cell metadata file"),
        (config['diet_bounds_file'], "Dietary bounds JSON"),
        (config['model_file'], "Metabolic model (iMM1415.json)"),
    ]
    
    all_valid = True
    for filepath, desc in required_files:
        if not validate_file_exists(filepath, desc, logger):
            all_valid = False
    
    # Validate condition mapping JSON
    try:
        json.loads(config['condition_mapping'])
        logger.info("  [OK] Condition mapping JSON is valid")
    except json.JSONDecodeError as e:
        logger.error(f"  [X] Invalid condition mapping JSON: {e}")
        all_valid = False
    
    return all_valid


def validate_rq2_rq3_inputs(config: Dict, logger: logging.Logger) -> bool:
    """Validate inputs for RQ2-RQ3 integration analysis."""
    logger.info("\nValidating RQ2-RQ3 integration inputs...")
    
    base_dir = config['rq2_base_dir']
    
    # RQ1 stats is in base_dir, but RQ3 outputs can be specified separately
    rq1_stats = config.get('rq1_pairwise_stats', 
                           os.path.join(base_dir, 'RQ1_multidataset_flux_pairwise_stats.csv'))
    rq3_stats = config.get('rq3_stats', 
                           os.path.join(base_dir, 'RQ3_statistical_tests.csv'))
    rq3_aggregation = config.get('rq3_aggregation', 
                                  os.path.join(base_dir, 'RQ3_aggregation_summary.csv'))
    
    required_files = [
        (rq1_stats, "RQ1 pairwise statistics"),
        (rq3_stats, "RQ3 statistical tests"),
        (rq3_aggregation, "RQ3 aggregation summary"),
    ]
    
    all_valid = True
    for filepath, desc in required_files:
        if not validate_file_exists(filepath, desc, logger):
            all_valid = False
    
    # -----------------------------------------------------------------------
    # Check per-strain RQ2 files.
    #
    # Actual layout on disk:
    #   {base_dir}/results_{strain}_{dataset_id}/flux_analysis/
    #       reaction_flux_comparison_extended.csv
    #
    # `dataset_id` is configurable (default: GSE182668) so the pipeline
    # remains usable if the GEO accession changes in future analyses.
    # -----------------------------------------------------------------------
    dataset_id = config.get('rq2_dataset_id', 'GSE182668')
    strains_list = config.get('strains', DEFAULT_STRAINS)
    logger.info(f"  Checking per-strain RQ2 files (dataset_id={dataset_id})...")
    strains_found = []
    strains_missing = []

    for strain in strains_list:
        strain_file = os.path.join(
            base_dir,
            f'results_{strain}_{dataset_id}',
            'flux_analysis',
            'reaction_flux_comparison_extended.csv'
        )
        if os.path.isfile(strain_file):
            strains_found.append(strain)
            logger.debug(f"    [OK] {strain}: {strain_file}")
        else:
            strains_missing.append(strain)
            logger.debug(f"    [!]  {strain}: {strain_file}")

    if strains_found:
        logger.info(f"  [OK] Found {len(strains_found)}/{len(strains_list)} strain files")
    if strains_missing:
        logger.warning(f"  [!] Missing strain files: {', '.join(strains_missing)}")
        logger.warning(
            f"      Expected layout: "
            f"{base_dir}/results_<strain>_{dataset_id}/flux_analysis/"
            f"reaction_flux_comparison_extended.csv"
        )
        # Not critical — pipeline runs with the strains that are present

    return all_valid and len(strains_found) > 0


def validate_rq1_rq2_rq3_inputs(config: Dict, logger: logging.Logger) -> bool:
    """Validate inputs for full integration analysis."""
    logger.info("\nValidating RQ1-RQ2-RQ3 integration inputs...")
    
    required_files = [
        (config['rq1_pairwise_stats'], "RQ1 pairwise statistics"),
        (config['rq3_stats'], "RQ3 statistical tests"),
        (config['rq3_aggregation'], "RQ3 aggregation summary"),
    ]
    
    all_valid = True
    for filepath, desc in required_files:
        if not validate_file_exists(filepath, desc, logger):
            all_valid = False
    
    return all_valid


# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

def run_command(cmd: List[str], description: str, logger: logging.Logger,
                cwd: Optional[str] = None) -> Tuple[bool, str]:
    """Run a command and capture output."""
    logger.info(f"\n{'='*80}")
    logger.info(f"EXECUTING: {description}")
    logger.info(f"{'='*80}")
    logger.debug(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=7200  # 2 hour timeout
        )
        
        # Log stdout
        if result.stdout:
            for line in result.stdout.split('\n'):
                if line.strip():
                    logger.info(f"  {line}")
        
        # Log stderr (may contain warnings)
        if result.stderr:
            for line in result.stderr.split('\n'):
                if line.strip():
                    if 'error' in line.lower():
                        logger.error(f"  {line}")
                    else:
                        logger.debug(f"  {line}")
        
        if result.returncode == 0:
            logger.info(f"[OK] {description} completed successfully")
            return True, result.stdout
        else:
            logger.error(f"[X] {description} failed with return code {result.returncode}")
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        logger.error(f"[X] {description} timed out after 2 hours")
        return False, "Timeout"
    except Exception as e:
        logger.error(f"[X] {description} failed with exception: {e}")
        return False, str(e)


def run_rq3_analysis(config: Dict, scripts_dir: str, logger: logging.Logger) -> bool:
    """Execute RQ3 single-cell E-Flux analysis."""
    
    script_path = os.path.join(scripts_dir, 'rq3_FINAL_COMPLETE_REVISED.py')
    
    
    if not os.path.isfile(script_path):
        logger.error(f"RQ3 script not found: {script_path}")
        return False
    
    # Build command
    cmd = [
        sys.executable, script_path,
        '--sc_data', config['sc_data'],
        '--sc_metadata', config['sc_metadata'],
        '--diet_bounds_file', config['diet_bounds_file'],
        '--condition_mapping', config['condition_mapping'],
        '--model', config['model_file'],
        '--objective', config['params']['objective'],
        '--eflux_quantile', str(config['params']['eflux_quantile']),
        '--eflux_floor', str(config['params']['eflux_floor']),
        '--eflux_cap', str(config['params']['eflux_cap']),
        '--normalization_strategy', config['params']['normalization_strategy'],
        '--baseline', config['params']['baseline_condition'],
        '--fold_change_threshold', str(config['params']['fold_change_threshold']),
        '--abs_change_threshold', str(config['params']['abs_change_threshold']),
        '--min_cells', str(config['params']['min_cells']),
        '--results_dir', config['output_dir'],
    ]
    
    # Add test conditions if specified
    if config['params'].get('test_conditions'):
        cmd.extend(['--test_conditions', config['params']['test_conditions']])
    
    # Add FVA option if enabled
    if config['params'].get('run_fva'):
        cmd.append('--run_fva')
        cmd.extend(['--fva_fraction', str(config['params']['fva_fraction'])])
    
    success, output = run_command(cmd, "RQ3: Single-cell E-Flux Analysis", logger)
    return success


def run_rq2_rq3_integration(config: Dict, scripts_dir: str, logger: logging.Logger) -> bool:
    """Execute RQ2-RQ3 multi-strain integration analysis."""

    # Try the ENHANCED version first (supports command-line args)
    script_path = os.path.join(scripts_dir, 'rq2_rq3_multi_strain_integration_ENHANCED.py')
    if not os.path.isfile(script_path):
        # Fall back to original
        script_path = os.path.join(scripts_dir, 'rq2_rq3_multi_strain_integration_FIXED_V2.py')

    if not os.path.isfile(script_path):
        logger.error(f"RQ2-RQ3 integration script not found: {script_path}")
        return False

    base_dir   = config['rq2_base_dir']
    dataset_id = config.get('rq2_dataset_id', 'GSE182668')
    params     = config.get('params', {})
    strains    = config.get('strains', DEFAULT_STRAINS)

    # ------------------------------------------------------------------
    # Build the per-strain file paths using the real nested layout:
    #   {base_dir}/results_{strain}_{dataset_id}/flux_analysis/
    #       reaction_flux_comparison_extended.csv
    # Only include strains whose file is actually present on disk.
    # ------------------------------------------------------------------
    strain_files: Dict[str, str] = {}
    for strain in strains:
        fpath = os.path.join(
            base_dir,
            f'results_{strain}_{dataset_id}',
            'flux_analysis',
            'reaction_flux_comparison_extended.csv'
        )
        if os.path.isfile(fpath):
            strain_files[strain] = fpath
        else:
            logger.warning(f"  [!] Strain file not found, skipping: {strain} -> {fpath}")

    if not strain_files:
        logger.error("[X] No valid per-strain RQ2 files found — cannot run RQ2-RQ3 integration")
        return False

    logger.info(f"  Running with {len(strain_files)}/{len(strains)} strains: "
                f"{', '.join(strain_files.keys())}")

    # Build command with explicit paths
    cmd = [
        sys.executable, script_path,
        '--base_dir',        base_dir,
        '--rq1_stats',       config.get('rq1_pairwise_stats',
                                        os.path.join(base_dir, 'RQ1_multidataset_flux_pairwise_stats.csv')),
        '--rq3_stats',       config.get('rq3_stats'),
        '--rq3_aggregation', config.get('rq3_aggregation'),
        '--contribution_threshold', str(params.get('contribution_threshold', 0.15)),
        '--fdr_threshold',   str(params.get('fdr_threshold', 0.10)),
        '--bulk_comparison', params.get('bulk_comparison', 'WD_vs_SCD'),
        '--cellular_comparison', params.get('cellular_comparison', 'WesternDiet_vs_Chow'),
        # Pass only the strains that have files on disk
        '--strains',         ','.join(strain_files.keys()),
    ]

    # Pass each per-strain file path explicitly so the downstream script does
    # not have to reconstruct paths itself (avoids the same layout bug there).
    # Format: --strain_files "C57BL6J:/path/to/file.csv,AJ:/path/to/file.csv"
    strain_files_arg = ','.join(f'{s}:{p}' for s, p in strain_files.items())
    cmd.extend(['--strain_files', strain_files_arg])

    # Add output directory if specified
    if config.get('output_dir'):
        cmd.extend(['--output_dir', config['output_dir']])

    success, output = run_command(
        cmd,
        "RQ2-RQ3: Multi-Strain Integration Analysis",
        logger
    )
    return success


def run_rq1_rq2_rq3_integration(config: Dict, scripts_dir: str, logger: logging.Logger) -> bool:
    """Execute full RQ1-RQ2-RQ3 integration analysis."""
    
    script_path = os.path.join(scripts_dir, 'rq1_rq2_rq3_integration_analysis_REVISED.py')
    
    if not os.path.isfile(script_path):
        logger.error(f"Full integration script not found: {script_path}")
        return False
    
    # Build command
    cmd = [
        sys.executable, script_path,
        '--rq1_pairwise_stats', config['rq1_pairwise_stats'],
        '--rq3_stats', config['rq3_stats'],
        '--rq3_aggregation', config['rq3_aggregation'],
        '--bulk_comparison', config['params']['bulk_comparison'],
        '--cellular_comparison', config['params']['cellular_comparison'],
        '--fdr_threshold', str(config['params']['fdr_threshold']),
        '--output_dir', config['output_dir'],
    ]
    
    # Add literature abundance option if enabled
    if config['params'].get('use_literature_abundance'):
        cmd.append('--use_literature_abundance')
    
    success, output = run_command(
        cmd, 
        "RQ1-RQ2-RQ3: Full Integration Analysis", 
        logger
    )
    return success


# =============================================================================
# MANIFEST GENERATION
# =============================================================================

def generate_manifest(config: Dict, results: Dict, output_dir: str, logger: logging.Logger):
    """Generate a manifest file documenting the pipeline run."""
    
    manifest = {
        'pipeline_version': '1.0',
        'run_timestamp': datetime.now().isoformat(),
        'python_version': sys.version,
        'configuration': config,
        'execution_results': results,
        'output_directory': output_dir,
    }
    
    manifest_path = os.path.join(output_dir, 'pipeline_manifest.json')
    
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, default=str)
    
    logger.info(f"\n[OK] Pipeline manifest saved: {manifest_path}")
    
    # Also generate a human-readable summary
    summary_path = os.path.join(output_dir, 'pipeline_summary.txt')
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("MASTER RQ INTEGRATION PIPELINE - EXECUTION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Run Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output Directory: {output_dir}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("EXECUTION RESULTS\n")
        f.write("-" * 80 + "\n")
        
        for step, result in results.items():
            status = "[OK] SUCCESS" if result['success'] else "[X] FAILED"
            f.write(f"\n{step}:\n")
            f.write(f"  Status: {status}\n")
            if result.get('output_dir'):
                f.write(f"  Output: {result['output_dir']}\n")
            if result.get('error'):
                f.write(f"  Error: {result['error']}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    logger.info(f"[OK] Pipeline summary saved: {summary_path}")


# =============================================================================
# COPY OUTPUTS FOR DOWNSTREAM USE
# =============================================================================

def prepare_downstream_inputs(rq3_output_dir: str, rq2_base_dir: str, 
                              logger: logging.Logger) -> bool:
    """
    Copy RQ3 outputs to locations expected by downstream scripts.
    
    The RQ2-RQ3 integration script expects files in a specific location.
    This function copies RQ3 outputs to the expected paths.
    """
    logger.info("\nPreparing inputs for downstream scripts...")
    
    # Files to copy from RQ3 output
    rq3_files = [
        ('aggregation_summary.csv', 'RQ3_aggregation_summary.csv'),
        ('statistics/statistical_tests.csv', 'RQ3_statistical_tests.csv'),
    ]
    
    copied = 0
    for src_rel, dst_name in rq3_files:
        src_path = os.path.join(rq3_output_dir, src_rel)
        dst_path = os.path.join(rq2_base_dir, dst_name)
        
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)
            logger.info(f"  [OK] Copied: {src_rel} -> {dst_name}")
            copied += 1
        else:
            logger.warning(f"  [!] Not found: {src_path}")
    
    return copied > 0


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Master RQ Integration Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLE USAGE:
--------------
1. Run full pipeline with config file:
   python master_rq_integration_pipeline.py --config pipeline_config.json

2. Run only RQ3 analysis:
   python master_rq_integration_pipeline.py --run_rq3 \\
       --sc_data expression_matrix.csv \\
       --sc_metadata metadata.csv \\
       --diet_bounds diet_bounds.json \\
       --condition_mapping '{"Chow":"SCD","WesternDiet":"WD"}' \\
       --output_dir results_rq3

3. Run full integration (requires RQ3 outputs):
   python master_rq_integration_pipeline.py --run_all \\
       --config pipeline_config.json

CONFIGURATION FILE:
------------------
Create a JSON file with all paths and parameters. See --generate_template.
        """
    )
    
    # Run mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--run_all', action='store_true',
                           help='Run complete pipeline (RQ3 -> Full Integration -> RQ2-RQ3 multi-strain)')
    mode_group.add_argument('--run_rq3', action='store_true',
                           help='Run only RQ3 analysis')
    mode_group.add_argument('--run_rq2_rq3', action='store_true',
                           help='Run only RQ2-RQ3 integration')
    mode_group.add_argument('--run_full_integration', action='store_true',
                           help='Run only full RQ1-RQ2-RQ3 integration')
    mode_group.add_argument('--validate_only', action='store_true',
                           help='Only validate inputs, do not run')
    mode_group.add_argument('--generate_template', action='store_true',
                           help='Generate template configuration file')
    
    # Configuration
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration JSON file')
    
    # Script locations
    parser.add_argument('--scripts_dir', type=str, default=None,
                       help='Directory containing the RQ scripts (overrides config if supplied)')
    
    # RQ3 inputs (can override config)
    parser.add_argument('--sc_data', type=str, help='Single-cell expression matrix')
    parser.add_argument('--sc_metadata', type=str, help='Cell metadata file')
    parser.add_argument('--diet_bounds', type=str, help='Dietary bounds JSON')
    parser.add_argument('--condition_mapping', type=str, 
                       help='JSON string mapping conditions')
    parser.add_argument('--model', type=str, default=None,
                       help='Metabolic model file')
    
    # RQ2/Integration inputs
    parser.add_argument('--rq2_base_dir', type=str, 
                       default='Processing_outputs/Step_2_RQ2',
                       help='Base directory for RQ2 data')
    parser.add_argument('--rq1_pairwise_stats', type=str,
                       help='RQ1 pairwise statistics file')
    parser.add_argument('--rq3_stats', type=str,
                       help='RQ3 statistical tests file')
    parser.add_argument('--rq3_aggregation', type=str,
                       help='RQ3 aggregation summary file')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='Processing_outputs/Step_3_RQ3',
                       help='Master output directory')
    
    # Logging
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    return parser.parse_args()


def generate_config_template(output_path: str):
    """Generate a template configuration file."""
    
    template = {
        "pipeline_version": "1.0",
        "description": "Master RQ Integration Pipeline Configuration",
        
        "scripts": {
            "scripts_dir": "./scripts",
            "rq3_script": "rq3_FINAL_COMPLETE_REVISED.py",
            "rq2_rq3_script": "rq2_rq3_multi_strain_integration_FIXED_V2.py",
            "full_integration_script": "rq1_rq2_rq3_integration_analysis_REVISED.py"
        },
        
        "rq3": {
            "inputs": {
                "sc_data": "path/to/expression_matrix.csv",
                "sc_metadata": "path/to/metadata.csv",
                "diet_bounds_file": "path/to/diet_bounds.json",
                "condition_mapping": '{"Chow":"SCD","WesternDiet":"WD","HighFat":"HFD","Ketogenic":"KD"}',
                "model_file": "iMM1415.json"
            },
            "parameters": DEFAULT_CONFIG['rq3'],
            "output_dir": "results_rq3"
        },
        
        "rq2_rq3_integration": {
            "inputs": {
                "base_dir": "Fluxes_Data_multi_background_rq2",
                "strains": DEFAULT_STRAINS
            },
            "parameters": DEFAULT_CONFIG['rq2_rq3'],
            "output_dir": "results_rq2_rq3_integration"
        },
        
        "full_integration": {
            "inputs": {
                "rq1_pairwise_stats": "path/to/RQ1_multidataset_flux_pairwise_stats.csv",
                "rq3_stats": "results_rq3/statistics/statistical_tests.csv",
                "rq3_aggregation": "results_rq3/aggregation_summary.csv"
            },
            "parameters": DEFAULT_CONFIG['rq1_rq2_rq3'],
            "output_dir": "results_full_integration"
        },
        
        "output": {
            "master_output_dir": "master_pipeline_results"
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"[OK] Configuration template saved: {output_path}")
    print("\nEdit this file with your actual paths and parameters,")
    print("then run: python master_rq_integration_pipeline.py --config pipeline_config.json --run_all")


def load_config(config_path: str, args) -> Dict:
    """Load and merge configuration from file and command-line arguments."""
    
    if config_path and os.path.isfile(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Override with command-line arguments
    # RQ3 configuration
    rq3_config = config.get('rq3', {'inputs': {}, 'parameters': {}})
    
    if args.sc_data:
        rq3_config['inputs']['sc_data'] = args.sc_data
    if args.sc_metadata:
        rq3_config['inputs']['sc_metadata'] = args.sc_metadata
    if args.diet_bounds:
        rq3_config['inputs']['diet_bounds_file'] = args.diet_bounds
    if args.condition_mapping:
        rq3_config['inputs']['condition_mapping'] = args.condition_mapping
    if args.model:
        rq3_config['inputs']['model_file'] = args.model
    
    # Merge with defaults
    rq3_params = {**DEFAULT_CONFIG['rq3'], **rq3_config.get('parameters', {})}
    rq3_config['parameters'] = rq3_params
    config['rq3'] = rq3_config
    
    # RQ2-RQ3 configuration
    rq2_rq3_config = config.get('rq2_rq3_integration', {'inputs': {}, 'parameters': {}})
    if args.rq2_base_dir:
        rq2_rq3_config['inputs']['base_dir'] = args.rq2_base_dir
    
    rq2_rq3_params = {**DEFAULT_CONFIG['rq2_rq3'], **rq2_rq3_config.get('parameters', {})}
    rq2_rq3_config['parameters'] = rq2_rq3_params
    config['rq2_rq3_integration'] = rq2_rq3_config
    
    # Full integration configuration
    full_config = config.get('full_integration', {'inputs': {}, 'parameters': {}})
    if args.rq1_pairwise_stats:
        full_config['inputs']['rq1_pairwise_stats'] = args.rq1_pairwise_stats
    if args.rq3_stats:
        full_config['inputs']['rq3_stats'] = args.rq3_stats
    if args.rq3_aggregation:
        full_config['inputs']['rq3_aggregation'] = args.rq3_aggregation
    
    full_params = {**DEFAULT_CONFIG['rq1_rq2_rq3'], **full_config.get('parameters', {})}
    full_config['parameters'] = full_params
    config['full_integration'] = full_config
    
    # Output directory
    config['output'] = config.get('output', {})
    if args.output_dir:
        config['output']['master_output_dir'] = args.output_dir
    
    # Scripts directory — only override when the flag was explicitly supplied
    config['scripts'] = config.get('scripts', {})
    if args.scripts_dir is not None:
        config['scripts']['scripts_dir'] = args.scripts_dir
    # Guarantee a sensible fallback if still absent after both sources
    if not config['scripts'].get('scripts_dir'):
        config['scripts']['scripts_dir'] = 'scripts'
    
    return config


def main():
    """Main entry point."""
    args = parse_args()
    
    # Handle template generation
    if args.generate_template:
        generate_config_template('pipeline_config_template.json')
        return 0
    
    # Load configuration
    config = load_config(args.config, args)
    
    # Setup output directory
    master_output_dir = config.get('output', {}).get('master_output_dir', 'Step_3b_master_pipeline_results') # master_pipeline_results
    os.makedirs(master_output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(master_output_dir, args.log_level)
    
    logger.info("=" * 80)
    logger.info("MASTER RQ INTEGRATION PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Output directory: {master_output_dir}")
    
    # Determine which steps to run
    run_rq3 = args.run_all or args.run_rq3
    run_rq2_rq3 = args.run_all or args.run_rq2_rq3
    run_full = args.run_all or args.run_full_integration
    validate_only = args.validate_only
    
    if not any([run_rq3, run_rq2_rq3, run_full, validate_only]):
        logger.info("\nNo run mode specified. Use --run_all, --run_rq3, etc.")
        logger.info("Use --help for usage information.")
        return 0
    
    # Track results
    results = {}
    scripts_dir = config.get('scripts', {}).get('scripts_dir', '.')
    
    # ==========================================================================
    # STEP 1: RQ3 Analysis (CORE GENERATOR)
    # ==========================================================================
    
    if run_rq3:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: RQ3 SINGLE-CELL E-FLUX ANALYSIS (CORE GENERATOR)")
        logger.info("Creates cell-type-specific metabolic models and raw data")
        logger.info("=" * 80)
        
        rq3_inputs = config.get('rq3', {}).get('inputs', {})
        rq3_params = config.get('rq3', {}).get('parameters', {})
        rq3_output_dir = config.get('rq3', {}).get('output_dir', 
                                                    os.path.join(master_output_dir, 'rq3_results'))
        
        # Prepare config for validation and execution
        rq3_config = {
            'sc_data': rq3_inputs.get('sc_data', ''),
            'sc_metadata': rq3_inputs.get('sc_metadata', ''),
            'diet_bounds_file': rq3_inputs.get('diet_bounds_file', ''),
            'condition_mapping': rq3_inputs.get('condition_mapping', '{}'),
            'model_file': rq3_inputs.get('model_file', 'iMM1415.json'),
            'params': rq3_params,
            'output_dir': rq3_output_dir,
        }
        
        # Validate
        if validate_rq3_inputs(rq3_config, logger):
            if not validate_only:
                success = run_rq3_analysis(rq3_config, scripts_dir, logger)
                results['RQ3_Analysis'] = {
                    'success': success,
                    'output_dir': rq3_output_dir
                }
            else:
                logger.info("[OK] RQ3 inputs validated (dry run)")
                results['RQ3_Analysis'] = {'success': True, 'note': 'Validation only'}
        else:
            logger.error("[X] RQ3 input validation failed")
            results['RQ3_Analysis'] = {'success': False, 'error': 'Input validation failed'}
    
    # ==========================================================================
    # STEP 2: Full Integration (INTERPRETER)
    # ==========================================================================
    
    if run_full:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: FULL RQ1-RQ2-RQ3 INTEGRATION (INTERPRETER)")
        logger.info("Explains WHY bulk liver changes occur based on cellular data")
        logger.info("=" * 80)
        
        full_inputs = config.get('full_integration', {}).get('inputs', {})
        full_params = config.get('full_integration', {}).get('parameters', {})
        full_output_dir = config.get('full_integration', {}).get('output_dir',
                                                                  os.path.join(master_output_dir, 'full_integration'))
        
        # If RQ3 outputs exist, use them
        if run_rq3 and results.get('RQ3_Analysis', {}).get('success'):
            rq3_output = config.get('rq3', {}).get('output_dir',
                                                   os.path.join(master_output_dir, 'rq3_results'))
            full_inputs['rq3_stats'] = os.path.join(rq3_output, 'statistics', 'statistical_tests.csv')
            full_inputs['rq3_aggregation'] = os.path.join(rq3_output, 'aggregation_summary.csv')
        
        full_config = {
            'rq1_pairwise_stats': full_inputs.get('rq1_pairwise_stats', ''),
            'rq3_stats': full_inputs.get('rq3_stats', ''),
            'rq3_aggregation': full_inputs.get('rq3_aggregation', ''),
            'params': full_params,
            'output_dir': full_output_dir,
        }
        
        # Validate
        if validate_rq1_rq2_rq3_inputs(full_config, logger):
            if not validate_only:
                success = run_rq1_rq2_rq3_integration(full_config, scripts_dir, logger)
                results['Full_Integration'] = {
                    'success': success,
                    'output_dir': full_output_dir
                }
            else:
                logger.info("[OK] Full integration inputs validated (dry run)")
                results['Full_Integration'] = {'success': True, 'note': 'Validation only'}
        else:
            logger.error("[X] Full integration input validation failed")
            results['Full_Integration'] = {'success': False, 'error': 'Input validation failed'}
    
    # ==========================================================================
    # STEP 3: RQ2-RQ3 Integration (GENERALIZER)
    # ==========================================================================
    
    if run_rq2_rq3:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: RQ2-RQ3 MULTI-STRAIN INTEGRATION (GENERALIZER)")
        logger.info("Tests if findings hold across genetic diversity (9 strains)")
        logger.info("=" * 80)
        
        rq2_rq3_inputs = config.get('rq2_rq3_integration', {}).get('inputs', {})
        rq2_rq3_params = config.get('rq2_rq3_integration', {}).get('parameters', {})
        rq2_rq3_output_dir = config.get('rq2_rq3_integration', {}).get('output_dir')
        
        base_dir = rq2_rq3_inputs.get('base_dir', 'Resulting_Data')
        
        # Determine RQ3 output paths - prefer explicit config, then check RQ3 outputs
        rq3_stats = rq2_rq3_inputs.get('rq3_stats')
        rq3_aggregation = rq2_rq3_inputs.get('rq3_aggregation')
        
        # If not specified in config, try to use RQ3 outputs (same paths as full_integration)
        if not rq3_stats or not rq3_aggregation:
            # Check if RQ3 ran successfully and use its outputs
            if run_rq3 and results.get('RQ3_Analysis', {}).get('success'):
                rq3_output = config.get('rq3', {}).get('output_dir', 
                                                        os.path.join(master_output_dir, 'rq3_results'))
                if not rq3_stats:
                    rq3_stats = os.path.join(rq3_output, 'statistics', 'statistical_tests.csv')
                if not rq3_aggregation:
                    rq3_aggregation = os.path.join(rq3_output, 'aggregation_summary.csv')
            else:
                # Fall back to full_integration paths if available
                full_inputs = config.get('full_integration', {}).get('inputs', {})
                if not rq3_stats:
                    rq3_stats = full_inputs.get('rq3_stats', 
                                                os.path.join(base_dir, 'RQ3_statistical_tests.csv'))
                if not rq3_aggregation:
                    rq3_aggregation = full_inputs.get('rq3_aggregation',
                                                      os.path.join(base_dir, 'RQ3_aggregation_summary.csv'))
        
        rq2_config = {
            'rq2_base_dir':       base_dir,
            'rq2_dataset_id':     rq2_rq3_inputs.get('rq2_dataset_id', 'GSE182668'),
            'rq1_pairwise_stats': rq2_rq3_inputs.get('rq1_pairwise_stats',
                                                      os.path.join(base_dir, 'RQ1_multidataset_flux_pairwise_stats.csv')),
            'rq3_stats':          rq3_stats,
            'rq3_aggregation':    rq3_aggregation,
            'strains':            rq2_rq3_inputs.get('strains', DEFAULT_STRAINS),
            'params':             rq2_rq3_params,
            'output_dir':         rq2_rq3_output_dir,
        }
        
        # Validate
        if validate_rq2_rq3_inputs(rq2_config, logger):
            if not validate_only:
                success = run_rq2_rq3_integration(rq2_config, scripts_dir, logger)
                results['RQ2_RQ3_Integration'] = {
                    'success': success,
                    'output_dir': rq2_rq3_output_dir or os.path.join(base_dir, 
                                                                     'rq2_rq3_integration_results_FIXED_V2')
                }
            else:
                logger.info("[OK] RQ2-RQ3 inputs validated (dry run)")
                results['RQ2_RQ3_Integration'] = {'success': True, 'note': 'Validation only'}
        else:
            logger.error("[X] RQ2-RQ3 input validation failed")
            results['RQ2_RQ3_Integration'] = {'success': False, 'error': 'Input validation failed'}
    
    # ==========================================================================
    # GENERATE MANIFEST AND SUMMARY
    # ==========================================================================
    
    generate_manifest(config, results, master_output_dir, logger)
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE EXECUTION COMPLETE")
    logger.info("=" * 80)
    
    all_success = all(r.get('success', False) for r in results.values())
    
    for step, result in results.items():
        status = "[OK] SUCCESS" if result.get('success') else "[X] FAILED"
        logger.info(f"  {step}: {status}")
    
    logger.info(f"\nOverall Status: {'SUCCESS' if all_success else 'PARTIAL FAILURE'}")
    logger.info(f"Results Directory: {master_output_dir}")
    
    return 0 if all_success else 1


if __name__ == '__main__':
    sys.exit(main())
