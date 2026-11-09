import os
import subprocess
import sys

env = dict(os.environ)
official = os.path.exists("generate_data.py") and os.path.exists("mock_llm.py")
if official:
    gen = "generate_data.py"
    print("Using the OFFICIAL generate_data.py and mock_llm.py")
else:
    gen = os.path.join("standins", "generate_data.py")
    env["PYTHONPATH"] = os.path.abspath("standins") + os.pathsep + env.get("PYTHONPATH", "")
    print("=" * 70)
    print("WARNING: official generate_data.py / mock_llm.py not found in the repo root.")
    print("Running with STAND-INS (numbers will NOT match the brief's reference values).")
    print("Copy the official files into the root and re-run before submitting.")
    print("=" * 70)

STEPS = [(gen, True), ("part_a.py", True), ("validate_part_a.py", official),
         ("part_b.py", True), ("part_c.py", True), ("part_d.py", True), ("build_readme.py", True)]
for script, strict in STEPS:
    print(f"== {script}")
    r = subprocess.run([sys.executable, script], env=env)
    if r.returncode and strict:
        sys.exit(f"{script} failed (exit {r.returncode})")
    if r.returncode:
        print(f"(checkpoint mismatch ignored in stand-in mode: {script})")
print("\nPipeline finished. README.md and outputs/ are generated.")
if "--chat" in sys.argv:
    print("Starting the customer chat app ...")
    subprocess.run([sys.executable, "app.py"], env=env)
else:
    print("Tip: `python run_all.py --chat` (or `python app.py`) opens the customer chat app.")
