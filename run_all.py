"""Run the whole pipeline for one split protocol, in order.

    python run_all.py              # block split (default)
    python run_all.py random       # random split, steps 1-3 only (for the comparison)

Progress of each training run is written to results/<split>/logs/*.log.
"""
import os
import subprocess
import sys
import time

split = sys.argv[1] if len(sys.argv) > 1 else "block"
steps = ["01_explore.py", "02_baselines.py", "02_plots.py", "02b_leakage_check.py", "03_transfer.py", "03_plots.py"]
if split == "block":
    steps += ["04_robustness.py", "05_gradcam.py", "06_cross_dataset.py"]

env = {**os.environ, "DEFECT_SPLIT": split, "PYTHONIOENCODING": "utf-8", "PYTHONWARNINGS": "ignore"}
start = time.time()
for step in steps:
    print(f"\n=== [{split}] {step}  (elapsed {(time.time() - start) / 60:.0f} min)", flush=True)
    subprocess.run([sys.executable, "-u", step], env=env, check=True)
print(f"\nall done in {(time.time() - start) / 60:.0f} min")
