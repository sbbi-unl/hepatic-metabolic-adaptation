#!/usr/bin/env python3
"""
Metabolic Flux Analysis Pipeline
"""

import subprocess
import json
import yaml
import argparse
import sys
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple


class FluxPipeline:
    """Manages the complete flux analysis pipeline with reproducibility tracking."""
    
    def __init__(self, config: Dict, run_parallel: bool = False):
        self.config = config
        self.run_parallel = run_parallel
        self.manifest = {
            'pipeline_version': '1.0.0',
            'timestamp': datetime.now().isoformat(),
            'run_mode': 'parallel_comparison' if run_parallel else 'sequential',
            'stages': {},
            'parameters': config,
            'outputs': {}
        }
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging to file and console."""
        # Dynamically set log dir based on results_dir to ensure it stays in the project folder
        base_dir = Path(self.config.get('results_dir', 'Processing_outputs/Step_1_RQ1'))
        log_dir = base_dir / 'pipeline_logs'
        log_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'pipeline_{timestamp}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Pipeline initialized. Log file: {log_file}")
        self.manifest['log_file'] = str(log_file)
        
    def run_command(self, cmd: List[str], stage_name: str, cwd: Optional[Path] = None) -> bool:
        """Execute a command and log the results."""
        self.logger.info(f"=== Starting Stage: {stage_name} ===")
        self.logger.info(f"Command: {' '.join(cmd)}")
        
        start_time = datetime.now()
        
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(f"[OK] {stage_name} completed successfully ({duration:.2f}s)")
            
            if result.stdout:
                self.logger.debug(f"STDOUT:\n{result.stdout}")
            
            self.manifest['stages'][stage_name] = {
                'status': 'success',
                'duration_seconds': duration,
                'command': ' '.join(cmd),
                'timestamp': datetime.now().isoformat()
            }
            
            return True
            
        except subprocess.CalledProcessError as e:
            duration = (datetime.now() - start_time).total_seconds()
            
            self.logger.error(f"[FAIL] {stage_name} failed ({duration:.2f}s)")
            self.logger.error(f"Error code: {e.returncode}")
            self.logger.error(f"STDERR:\n{e.stderr}")
            
            self.manifest['stages'][stage_name] = {
                'status': 'failed',
                'duration_seconds': duration,
                'command': ' '.join(cmd),
                'error': str(e),
                'stderr': e.stderr,
                'timestamp': datetime.now().isoformat()
            }
            
            return False
    
    def run_flux_analysis(self, suffix: str = "") -> Optional[Path]:
        """Run the main E-Flux analysis."""
        stage_name = f"flux_analysis{suffix}"
        
        folder_name = suffix.lstrip('_') if suffix else 'main_analysis'
        results_dir = Path(self.config['results_dir']) / folder_name
        results_dir.mkdir(exist_ok=True, parents=True)
        
        self.logger.info(f"Running E-Flux analysis...")
        self.logger.info(f"  Input: {self.config['input_expression']}")
        self.logger.info(f"  Output: {results_dir}")
        
        cmd = [
            'python',
            self.config['flux_script'],
            self.config['input_expression'],
            '--model_file', self.config['model_file'],
            '--results_dir', str(results_dir),
            '--test_type', self.config.get('test_type', 't-test'),
            '--diet_bounds_json', self.config['diet_bounds_json'],
            '--eflux_quantile', str(self.config['eflux_quantile']),
            '--eflux_floor', str(self.config['eflux_floor']),
            '--eflux_cap', str(self.config['eflux_cap']),
            '--objective_id', self.config['objective_id'],
            '--objective_sense', self.config['objective_sense'],
            '--transporter_strategy', self.config['transporter_strategy'],
            '--edge_abs_diff_threshold', str(self.config['edge_abs_diff_threshold']),
            '--mapping_file', self.config['mapping_file']
        ]
        
        if self.config.get('write_replicates_long', True):
            cmd.append('--write_replicates_long')
        
        if self.config.get('rank_product_for'):
            cmd.extend(['--rank_product_for', self.config['rank_product_for']])
        
        if self.config.get('no_fva', False):
            cmd.append('--no_fva')
            self.logger.info("  FVA: Disabled (--no_fva)")
        else:
            self.logger.info("  FVA: Enabled")
        
        if self.config.get('aggregate', False):
            cmd.append('--aggregate')
            self.logger.info("  Analysis mode: AGGREGATE (averaging replicates within groups)")
        else:
            self.logger.info("  Analysis mode: REPLICATE-LEVEL (individual replicates)")
        
        success = self.run_command(cmd, stage_name)
        
        if success:
            results_dir_path = results_dir
            
            possible_paths = [
                results_dir_path / 'combined_reaction_flux_comparison_extended.csv',
                results_dir_path / 'flux_analysis' / 'combined_reaction_flux_comparison_extended.csv',
                results_dir_path / 'flux_analysis' / 'reaction_flux_comparison_extended.csv', 
                results_dir_path / 'stats_comparison' / 'combined_reaction_flux_comparison_extended.csv',
                results_dir_path / 'reaction_flux_comparison_extended.csv', 
            ]
            
            flux_output = None
            for path in possible_paths:
                if path.exists():
                    flux_output = path
                    break
            
            if flux_output:
                self.manifest['outputs'][f'flux_comparison{suffix}'] = str(flux_output)
                self.logger.info(f"  [OK] Flux output created: {flux_output}")
                return flux_output 
            else:
                self.logger.error(f"  [FAIL] Expected output not found in any of:")
                for path in possible_paths:
                    self.logger.error(f"    - {path}")
                self.logger.error(f"  Available files in {results_dir_path}:")
                if results_dir_path.exists():
                    for item in sorted(results_dir_path.iterdir()):
                        if item.is_file():
                            self.logger.error(f"    - {item.name}")
                return None
        else:
            return None
    
    def run_batch_correction(self, input_file: Path, suffix: str = "") -> Optional[Path]:
        """Run batch correction on flux data."""
        if not self.config.get('batch_script'):
            self.logger.warning("No batch correction script specified, skipping batch correction")
            return input_file
        
        stage_name = f"batch_correction{suffix}"
        output_file = input_file.parent / f"{input_file.stem}_batch_corrected.csv"
        
        batch_outdir = input_file.parents[1] / "batch_correction_plots"
        batch_outdir.mkdir(exist_ok=True, parents=True)
        
        self.logger.info(f"Running batch correction...")
        self.logger.info(f"  Input: {input_file}")
        self.logger.info(f"  Output: {output_file}")
        
        cmd = [
            'python',
            self.config['batch_script'],
            '--input', str(input_file),
            '--output_csv', str(output_file),
            '--outdir', str(batch_outdir)
        ]
        
        success = self.run_command(cmd, stage_name)
        
        if success and output_file.exists():
            self.manifest['outputs'][f'batch_corrected{suffix}'] = str(output_file)
            self.logger.info(f"  [OK] Batch-corrected file created: {output_file}")
            return output_file
        else:
            self.logger.error(f"  [FAIL] Batch correction failed or output not found")
            return None
    
    def run_combined_analysis(self, input_file: Path, suffix: str = "") -> bool:
        """Run the combined analysis (PCA, PERMANOVA, visualizations)."""
        if not self.config.get('combined_script'):
            self.logger.warning("No combined analysis script specified, skipping combined analysis")
            return True
        
        stage_name = f"combined_analysis{suffix}"
        
        analysis_dir = input_file.parents[1] / "combined_analysis_plots"
        analysis_dir.mkdir(exist_ok=True, parents=True)
        
        self.logger.info(f"Running combined analysis (PCA, PERMANOVA, visuals)...")
        self.logger.info(f"  Input: {input_file}")
        
        cmd = [
            'python',
            self.config['combined_script'],
            '--input', str(input_file),
            '--output_dir', str(analysis_dir)
        ]
        
        success = self.run_command(cmd, stage_name)
        
        if success:
            self.logger.info(f"  [OK] Combined analysis completed")
            if analysis_dir.exists():
                self.manifest['outputs'][f'combined_analysis_dir{suffix}'] = str(analysis_dir)
        
        return success

    def run_comprehensive_analysis(self, input_file: Path) -> bool:
        """Run the final comprehensive flux analysis."""
        if not self.config.get('comprehensive_script'):
            self.logger.warning("No comprehensive analysis script specified, skipping comprehensive analysis")
            return True
            
        stage_name = "comprehensive_analysis"
        
        # Explicitly define Comprehensive Analysis output dir inside the main results_dir
        comp_dir = Path(self.config['results_dir']) / "Comprehensive_Analysis"
        comp_dir.mkdir(exist_ok=True, parents=True)
        
        self.logger.info(f"Running comprehensive flux analysis...")
        self.logger.info(f"  Input: {input_file}")
        self.logger.info(f"  Output: {comp_dir}")
        
        cmd = [
            'python',
            self.config['comprehensive_script'],
            '--input', str(input_file),
            '--output', str(comp_dir)
        ]
        
        success = self.run_command(cmd, stage_name)
        
        if success:
            self.logger.info(f"  [OK] Comprehensive analysis completed")
            self.manifest['outputs']['comprehensive_analysis_dir'] = str(comp_dir)
            
        return success
    
    def run_analysis_branch(self, suffix: str = "", run_batch_correction: bool = True) -> Tuple[bool, Optional[Path]]:
        """Run a complete analysis branch, returning success and final processed CSV."""
        # Step 1: Run E-Flux analysis
        flux_output = self.run_flux_analysis(suffix)
        if not flux_output:
            self.logger.error(f"Branch {suffix} failed at E-Flux analysis stage")
            return False, None
        
        # Step 2: Optionally run batch correction
        if run_batch_correction:
            corrected_output = self.run_batch_correction(flux_output, suffix)
            if not corrected_output:
                self.logger.error(f"Branch {suffix} failed at batch correction stage")
                return False, None
            analysis_input = corrected_output
        else:
            analysis_input = flux_output
        
        # Step 3: Run combined analysis
        success = self.run_combined_analysis(analysis_input, suffix)
        if not success:
            self.logger.error(f"Branch {suffix} failed at combined analysis stage")
            return False, None
        
        self.logger.info(f"Branch {suffix} completed successfully")
        return True, analysis_input
    
    def run_pipeline(self) -> bool:
        """Run the complete pipeline."""
        overall_start = datetime.now()
        
        self.logger.info("=" * 80)
        self.logger.info("METABOLIC FLUX ANALYSIS PIPELINE")
        self.logger.info("=" * 80)
        self.logger.info(f"Start time: {overall_start.isoformat()}")
        self.logger.info(f"Input file: {self.config['input_expression']}")
        self.logger.info(f"Results directory: {self.config['results_dir']}")
        
        if self.config.get('aggregate', False):
            self.logger.info("Analysis mode: AGGREGATE (replicates averaged within groups)")
        else:
            self.logger.info("Analysis mode: REPLICATE-LEVEL (individual replicates)")
        
        self.logger.info("=" * 80 + "\n")
        
        final_corrected_file = None

        if self.run_parallel:
            self.logger.info("Mode: Parallel comparison (corrected vs uncorrected)")
            
            self.logger.info("\n" + "="*80)
            self.logger.info("BRANCH 1/2: UNCORRECTED ANALYSIS")
            self.logger.info("="*80 + "\n")
            success_uncorrected, _ = self.run_analysis_branch('_uncorrected', False)
            
            self.logger.info("\n" + "="*80)
            self.logger.info("BRANCH 2/2: BATCH-CORRECTED ANALYSIS")
            self.logger.info("="*80 + "\n")
            success_corrected, final_corrected_file = self.run_analysis_branch('_batch_corrected', True)
            
            success = success_uncorrected and success_corrected
            
            if success:
                self.logger.info("\n" + "="*80)
                self.logger.info("PARALLEL COMPARISON COMPLETED SUCCESSFULLY")
                self.logger.info("="*80)
            else:
                self.logger.error("\n" + "="*80)
                self.logger.error("PARALLEL COMPARISON FAILED")
                self.logger.error("="*80)
        else:
            self.logger.info("Mode: Sequential (batch-corrected only)")
            success, final_corrected_file = self.run_analysis_branch('', True)

        # =====================================================================
        # FINAL STEP: COMPREHENSIVE ANALYSIS
        # =====================================================================
        if success and final_corrected_file:
            self.logger.info("\n" + "="*80)
            self.logger.info("FINAL STEP: COMPREHENSIVE ANALYSIS")
            self.logger.info("="*80 + "\n")
            
            comp_success = self.run_comprehensive_analysis(final_corrected_file)
            success = success and comp_success

        total_duration = (datetime.now() - overall_start).total_seconds()
        self.manifest['total_duration_seconds'] = total_duration
        self.manifest['overall_status'] = 'success' if success else 'failed'
        
        self.save_manifest()
        
        self.logger.info("\n" + "="*80)
        self.logger.info(f"Pipeline completed in {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
        self.logger.info(f"Status: {'SUCCESS' if success else 'FAILED'}")
        self.logger.info("="*80 + "\n")
        
        return success
    
    def save_manifest(self):
        """Save the manifest to both YAML and JSON formats."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        manifest_dir = Path(self.config['results_dir']) / 'pipeline_manifests'
        manifest_dir.mkdir(exist_ok=True, parents=True)
        
        yaml_file = manifest_dir / f'manifest_{timestamp}.yaml'
        with open(yaml_file, 'w') as f:
            yaml.dump(self.manifest, f, default_flow_style=False, sort_keys=False)
        
        json_file = manifest_dir / f'manifest_{timestamp}.json'
        with open(json_file, 'w') as f:
            json.dump(self.manifest, f, indent=2)
        
        self.manifest['manifest_file'] = str(yaml_file)
        self.logger.info(f"Manifest saved: {yaml_file} and {json_file}")


def create_default_config() -> Dict:
    """Create default configuration dictionary."""
    return {
        # Input files
        'input_expression': 'data/GSEMERGED_SCD_HFD_KD_WD_gene_expression.csv',
        'model_file': 'data/iMM1415.json',
        'diet_bounds_json': 'data/expanded_diet_bounds_flat.json',
        'mapping_file': 'data/mouse_entrez_to_symbol.csv',
        
        # Scripts  
        'flux_script': 'scripts/map_fixv5_multigroupsv8_layered_manuscript_run.py',
        'batch_script': 'scripts/batch_correct_flux_by_dataset_3d.py',
        'combined_script': 'scripts/combined_analysis_interactive_generic.py',
        'comprehensive_script': 'scripts/comprehensive_flux_analysis.py',
        
        # E-Flux parameters
        'results_dir': 'results_Biomassfl01',
        'test_type': 'mann-whitney',
        'eflux_quantile': 0.95,
        'eflux_floor': 0.1,
        'eflux_cap': 1000,
        'objective_id': 'BIOMASS_mm_1_no_glygln',
        'objective_sense': 'max',
        'transporter_strategy': 'either',
        'edge_abs_diff_threshold': 0.2,
        'rank_product_for': 'HFD,KD,WD',
        
        # Analysis flags
        'write_replicates_long': True,
        'no_fva': False,
        'aggregate': False,  
        
        # Pipeline settings
        'log_dir': 'pipeline_logs' 
    }


def main():
    parser = argparse.ArgumentParser(
        description='Run complete metabolic flux analysis pipeline with reproducibility tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--config', type=str, help='Path to YAML configuration file (overrides defaults)')
    parser.add_argument('--parallel', action='store_true', help='Run both corrected and uncorrected branches in parallel for comparison')
    parser.add_argument('--generate_config', action='store_true', help='Generate default configuration file and exit')
    parser.add_argument('--input_expression', type=str)
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--results_dir', type=str)
    parser.add_argument('--eflux_quantile', type=float)
    parser.add_argument('--eflux_floor', type=float)
    parser.add_argument('--eflux_cap', type=float)
    parser.add_argument('--objective_id', type=str)
    parser.add_argument('--no_fva', action='store_true', help='Disable flux variability analysis')
    parser.add_argument('--write_replicates_long', action='store_true', default=True, help='Write per-replicate flux values (default: True)')
    parser.add_argument('--rank_product_for', type=str, help='Comma-separated list of diets for rank product analysis')
    parser.add_argument('--aggregate', action='store_true', help='Aggregate replicates within each group')
    
    args = parser.parse_args()
    
    if args.generate_config:
        config = create_default_config()
        print(yaml.dump(config, default_flow_style=False, sort_keys=False))
        return 0
    
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = create_default_config()
    
    for key, value in vars(args).items():
        if value is not None and key not in ['config', 'parallel', 'generate_config', 'no_fva', 'aggregate', 'write_replicates_long']:
            config[key] = value
    
    if args.no_fva:
        config['no_fva'] = True
    if hasattr(args, 'write_replicates_long'):
        config['write_replicates_long'] = args.write_replicates_long
    if args.aggregate:
        config['aggregate'] = True
    
    pipeline = FluxPipeline(config, run_parallel=args.parallel)
    success = pipeline.run_pipeline()
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())