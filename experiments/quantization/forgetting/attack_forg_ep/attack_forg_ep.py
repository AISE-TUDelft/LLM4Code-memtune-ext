#!/usr/bin/env python3
"""
Sequential execution of forgetting attack experiments across all epochs.
Runs all quantization and epoch combinations in the correct order.

IMPORTANT NOTE: bf16_ep0.py should always be the first one to be run because it creates the final .parquet file
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
        'bf16_ep0.py',   # Creates initial parquet
        'bf16_ep1.py',   # Adds bf16_ep1 column
        'bf16_ep2.py',   # Adds bf16_ep2 column 
        'bf16_ep3.py',   # Adds bf16_ep3 column
        'i8_ep0.py',     # Adds i8_ep0 column
        'i8_ep1.py',     # Adds i8_ep1 column
        'i8_ep2.py',     # Adds i8_ep2 column
        'i8_ep3.py',     # Adds i8_ep3 column
        'i4_ep0.py',     # Adds i4_ep0 column
        'i4_ep1.py',     # Adds i4_ep1 column
        'i4_ep2.py',     # Adds i4_ep2 column
        'i4_ep3.py'      # Adds i4_ep3 column
    ]
    
    print("🚀 Starting forgetting attack experiment suite (all epochs)")
    print(f"Total scripts to run: {len(scripts)}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Experiment order: bf16 (ep0→ep3) → i8 (ep0→ep3) → i4 (ep0→ep3)")
    
    total_start_time = time.time()
    
    for i, script in enumerate(scripts, 1):
        print(f"\n📋 Progress: {i}/{len(scripts)}")
        
        success = run_script(script)
        
        if not success:
            print(f"\n🛑 Stopping execution due to error in {script}")
            print("Fix the error and restart from this script.")
            return False
    
    total_elapsed = time.time() - total_start_time
    print(f"\n🎉 All experiments completed successfully!")
    print(f"Total execution time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    print(f"Results saved in: attack_forg_ep.parquet")
    print("\nFinal dataset contains columns:")
    print("  - bf16_ep0, bf16_ep1, bf16_ep2, bf16_ep3")
    print("  - i8_ep0, i8_ep1, i8_ep2, i8_ep3") 
    print("  - i4_ep0, i4_ep1, i4_ep2, i4_ep3")
    print("  - All corresponding evaluation metrics (*_em, *_bleu, *_meteor, *_rougeL)")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
