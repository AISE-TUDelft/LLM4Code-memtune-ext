#!/usr/bin/env python3
"""
Sequential execution of HumanEval-X Java generation for all 8 models.
Each model runs as a separate subprocess so the GPU is fully released between runs.

Run from the repo root:
    python benchmark/generate_humaneval_java.py

Order:
    StarCoder2-3B  (base → ep3)
    StarCoder2-7B  (base → ep3)
    StarCoder2-15B (base → ep3)
    Mellum-4B      (base → ep3)
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

BENCHMARK_DIR = Path(__file__).parent

SCRIPTS = [
    'sc2_3b_base.py',
    'sc2_3b_ep3.py',
    'sc2_7b_base.py',
    'sc2_7b_ep3.py',
    'sc2_15b_base.py',
    'sc2_15b_ep3.py',
    'mellum_4b_base.py',
    'mellum_4b_ep3.py',
]


def run_script(script_name):
    script_path = BENCHMARK_DIR / script_name

    print(f"\n{'='*60}")
    print(f"Starting {script_name} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            check=True,
        )

        elapsed = time.time() - start_time
        print(f"\nCompleted {script_name} in {elapsed:.1f}s")

        if result.stdout:
            lines = result.stdout.strip().split('\n')
            print("Last output:")
            for line in lines[-3:]:
                print(f"  {line}")

        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\nError in {script_name} after {elapsed:.1f}s")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print("STDOUT:")
            print(e.stdout)
        if e.stderr:
            print("STDERR:")
            print(e.stderr)
        return False


def main():
    print("Starting HumanEval-X Java generation — all 8 models")
    print(f"Total scripts: {len(SCRIPTS)}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    total_start = time.time()

    for i, script in enumerate(SCRIPTS, 1):
        print(f"\nProgress: {i}/{len(SCRIPTS)}")
        success = run_script(script)

        if not success:
            print(f"\nStopping execution due to error in {script}")
            print("Fix the error and restart from this script.")
            return False

    total_elapsed = time.time() - total_start
    print(f"\nAll experiments completed successfully!")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    OUTPUT_NAMES = [
        "starcoder2-3b",
        "StarCoder2Java-3b_ep3",
        "starcoder2-7b",
        "StarCoder2Java-7b_ep3",
        "starcoder2-15b",
        "StarCoder2Java-15b_ep3",
        "Mellum-4b-base",
        "MellumJava_ep3",
    ]
    print("\nResults saved in:")
    for name in OUTPUT_NAMES:
        print(f"  benchmark/generations/{name}/generations.jsonl")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)