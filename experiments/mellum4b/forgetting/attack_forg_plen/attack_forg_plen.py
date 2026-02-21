#!/usr/bin/env python3
"""
Sequential execution of forgetting attack experiments across prefix lengths.
Runs all epoch combinations in the correct order.

IMPORTANT NOTE: me4b_ep0.py should always be the first one to be run because it creates the final .parquet file
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
        'me4b_ep0.py',   # Creates initial parquet
        'me4b_ep3.py',   # Adds me4b_ep3 columns
    ]

    print("🚀 Starting forgetting attack experiment suite (prefix lengths)")
    print(f"Total scripts to run: {len(scripts)}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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
    print(f"Results saved in: attack_forg_plen.parquet")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
