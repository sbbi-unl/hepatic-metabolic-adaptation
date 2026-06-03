#!/usr/bin/env python3
"""
WRAPPER SCRIPT: HIERARCHICAL ANALYSIS PIPELINE
==============================================
Runs the bulk validation, hierarchical attribution, and figure generation
scripts sequentially.
"""

import os
import sys
import subprocess

def run_script(script_path):
    """Executes a python script and checks for errors."""
    print(f"\n{'='*80}")
    print(f"🚀 RUNNING: {script_path}")
    print(f"{'='*80}\n")
    
    try:
        # Run the script and stream the output to the console
        result = subprocess.run([sys.executable, script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR: {script_path} failed with exit code {e.returncode}.")
        print("Pipeline halted.")
        sys.exit(e.returncode)

def main():
    # Define the scripts to run in order
    # Assuming this wrapper is run from the root project folder
    scripts = [
        "scripts/bulk_validation_LOCAL.py",
        "scripts/hierarchical_attribution_LOCAL.py",
        "scripts/hierarchical_figures_LOCAL.py"
    ]
    
    # Ensure all scripts exist before starting
    for script in scripts:
        if not os.path.isfile(script):
            print(f"❌ ERROR: Cannot find '{script}'.")
            print("Make sure you are running this from your main project folder (e.g., 'python scripts/run_hierarchical_analysis.py')")
            sys.exit(1)
            
    # Pre-create the output directory
    output_dir = "Processing_outputs/Step_3_RQ3/Hierarchical_Analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Execute the pipeline
    for script in scripts:
        run_script(script)
        
    print(f"\n{'='*80}")
    print(f"✅ PIPELINE COMPLETE: All hierarchical analyses finished successfully!")
    print(f"Outputs are available in: {output_dir}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()