import sys
import os
import subprocess
from pathlib import Path
from typing import List

def run_experiment(args: List[str]) -> subprocess.CompletedProcess:
    """Runs train_autism.py with the given arguments list using subprocess.
    
    Ensures environment variables like PYTHONPATH are set up correctly.
    """
    python_exe = sys.executable
    cmd = [python_exe, "-m", "training.train_autism"] + args
    
    env = os.environ.copy()
    root_dir = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = root_dir + (os.pathsep + env.get("PYTHONPATH", "")) if env.get("PYTHONPATH") else root_dir
    env["PYTHONIOENCODING"] = "utf-8"
    
    print(f"\n>>> Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    return result

if __name__ == "__main__":
    # Forward arguments directly if called from CLI
    sys.exit(run_experiment(sys.argv[1:]).returncode)
