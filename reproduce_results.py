import os
import subprocess
import sys

def run_command(command):
    print(f"Executing: {command}")
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        sys.exit(1)

def main():
    print("=== Micro-Tensor Reproducibility Suite ===")
    
    # 1. Install dependencies
    print("\n[1/3] Updating requirements...")
    run_command("pip install -r requirements.txt")
    
    # 2. Run Moons Experiment
    print("\n[2/3] Running Moons Training (Adam)...")
    run_command("python test_moons.py")
    
    # 3. Run Benchmark
    print("\n[3/3] Running Optimizer Benchmarking...")
    run_command("python benchmarking.py")
    
    print("\n=== Success! All visual assets generated ===")
    print("Files created: moon_boundary_adam.png, optimizer_comparison.png")

if __name__ == "__main__":
    main()
