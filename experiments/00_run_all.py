import subprocess
import sys
import time
from pathlib import Path

# Set to None to run every .py file in THIS directory, or provide your own list.
# The glob is not recursive, so anything under archive/ is skipped -- see archive/README.md.
scripts = [
    "26_what_remains_after_masking.py",
    "27_interaction_at_width_64.py"
    ]
scripts = None

here = Path(__file__).parent
this = Path(__file__).name

if scripts is None:
    scripts = sorted(f.name for f in here.glob("*.py") if f.name != this)

results = []

for script in scripts:
    print(f"\nRunning {script}...")

    start = time.perf_counter()

    try:
        r = subprocess.run(
            [sys.executable, script],
            cwd=here,
        )
        status = "OK" if r.returncode == 0 else f"FAILED ({r.returncode})"

    except Exception as e:
        status = f"ERROR ({e})"

    elapsed = time.perf_counter() - start
    results.append((script, status, elapsed))

print("\nSummary")
print("-" * 40)
for script, status, elapsed in results:
    print(f"{script:<20} {status:<15} {elapsed:.1f}s")