#!/usr/bin/env python3
"""
Orchestrator script for LoRA forgetting attack experiments across epochs.
Runs experiments for epochs 1, 2, and 3 sequentially.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_experiment(script_name):
    """Run a single experiment script and handle errors."""
    print(f"\n{'='*60}")
    print(f"Starting experiment: {script_name}")
    print(f"{'='*60}")

    try:
        # Run the script and capture output
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )

        # Print the output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"Warnings/Info: {result.stderr}")

        print(f"✅ {script_name} completed successfully!")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {script_name}")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error running {script_name}: {e}")
        return False

def main():
    """Main orchestrator function."""
    print("🚀 Starting LoRA Forgetting Attack Experiments (attack_forg_ep)")
    print("This will run experiments for epochs 1, 2, and 3 sequentially.")

    # Define the experiment sequence
    experiments = [
        "7b_lo_ep1.py",  # Creates initial parquet file
        "7b_lo_ep2.py",  # Merges with existing file
        "7b_lo_ep3.py"   # Merges with existing file
    ]

    # Check if all scripts exist
    for script in experiments:
        if not os.path.exists(script):
            print(f"❌ Error: {script} not found in current directory")
            sys.exit(1)

    # Run experiments sequentially
    completed_experiments = []

    for i, experiment in enumerate(experiments, 1):
        print(f"\n📊 Running experiment {i}/{len(experiments)}: {experiment}")

        success = run_experiment(experiment)

        if success:
            completed_experiments.append(experiment)
        else:
            print(f"\n💥 Experiment {experiment} failed!")
            print(f"Completed experiments so far: {completed_experiments}")
            print(f"To restart from this point, you can run: python {experiment}")
            sys.exit(1)

    # All experiments completed successfully
    print(f"\n🎉 All experiments completed successfully!")
    print(f"{'='*60}")
    print("Summary:")
    for exp in completed_experiments:
        print(f"  ✅ {exp}")
    print(f"\nResults saved to: attack_forg_ep.parquet")
    print("You can now analyze the forgetting results using the generated parquet file.")

if __name__ == "__main__":
    main()