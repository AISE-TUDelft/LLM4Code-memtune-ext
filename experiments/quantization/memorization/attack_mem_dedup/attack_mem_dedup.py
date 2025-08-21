#!/usr/bin/env python3
"""
Sequential execution of memorization attack experiments across all deduplication levels.
Runs all quantization and epoch combinations for deduplication attack experiments.

IMPORTANT NOTE: bf16_ep0.py MUST be run first as it creates the initial parquet files.
All other scripts depend on the existence of these files.
"""

import subprocess
import sys
import time
from datetime import datetime

def run_script(script_name):
    """Run a Python script and handle errors."""
    print(f"\n{'='*60}")
    print(f"Starting {script_name} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name], 
            capture_output=True, 
            text=True,
            check=True
        )
        
        elapsed = time.time() - start_time
        print(f"\n✅ Completed {script_name} successfully in {elapsed:.1f}s")
        
        # Print last few lines of output for confirmation
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            print("Last output:")
            for line in lines[-3:]:
                print(f"  {line}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Error in {script_name} after {elapsed:.1f}s")
        print(f"Return code: {e.returncode}")
        
        if e.stdout:
            print("STDOUT:")
            print(e.stdout)
        if e.stderr:
            print("STDERR:")
            print(e.stderr)
        
        return False

def main():
    """Run all experiments in sequence."""
    
    scripts = [
        'bf16_ep0.py',   # Creates initial 4 parquet files (MUST BE FIRST)
        'bf16_ep3.py',   # Adds bf16_ep3 columns to all 4 files
        'i8_ep0.py',     # Adds i8_ep0 columns to all 4 files
        'i8_ep3.py',     # Adds i8_ep3 columns to all 4 files
        'i4_ep0.py',     # Adds i4_ep0 columns to all 4 files
        'i4_ep3.py'      # Adds i4_ep3 columns to all 4 files
    ]
    
    print("🚀 Starting memorization attack deduplication experiment suite")
    print(f"Total scripts to run: {len(scripts)}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nExperiment design:")
    print("  - 4 deduplication levels: d1, d2, d3, dg3")
    print("  - 3 precisions: bf16, i8, i4")
    print("  - 2 epochs: ep0 (baseline), ep3 (fine-tuned)")
    print("  - 4 output files: attack_mem_dedup_1.parquet, attack_mem_dedup_2.parquet,")
    print("                   attack_mem_dedup_3.parquet, attack_mem_dedup_g3.parquet")
    
    total_start_time = time.time()
    
    for i, script in enumerate(scripts, 1):
        print(f"\n📋 Progress: {i}/{len(scripts)}")
        
        # Special handling for the first script
        if i == 1:
            print("⚠️  CRITICAL: Running bf16_ep0.py - this creates the base parquet files")
            print("   All subsequent scripts depend on these files existing")
        
        success = run_script(script)
        
        if not success:
            print(f"\n🛑 Stopping execution due to error in {script}")
            print("Fix the error and restart from this script.")
            
            # Provide restart guidance
            if i == 1:
                print("\n💡 RESTART GUIDANCE:")
                print("   Since bf16_ep0.py failed, restart from the beginning")
            else:
                print(f"\n💡 RESTART GUIDANCE:")
                print(f"   bf16_ep0.py has completed successfully")
                print(f"   You can restart from {script} once the issue is fixed")
            
            return False
    
    total_elapsed = time.time() - total_start_time
    print(f"\n🎉 All deduplication experiments completed successfully!")
    print(f"Total execution time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    
    print(f"\n📊 Final dataset summary:")
    print("  4 output files created:")
    print("    - attack_mem_dedup_1.parquet (d1 split)")
    print("    - attack_mem_dedup_2.parquet (d2 split)")
    print("    - attack_mem_dedup_3.parquet (d3 split)")
    print("    - attack_mem_dedup_g3.parquet (dg3 split)")
    
    print("\n  Each file contains columns:")
    print("    - bf16_ep0_d[X], bf16_ep3_d[X]")
    print("    - i8_ep0_d[X], i8_ep3_d[X]")
    print("    - i4_ep0_d[X], i4_ep3_d[X]")
    print("    - All corresponding evaluation metrics (*_em, *_bleu, *_meteor, *_rougeL)")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)