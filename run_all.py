"""
run_all.py  -  Run the full pipeline end to end.
Executes EDA -> modelling -> report generation in order.

Run:  python run_all.py
"""
import subprocess
import sys

STEPS = ["01_eda.py", "02_modeling.py", "03_generate_report.py"]

for step in STEPS:
    print(f"\n{'=' * 60}\nRunning {step}\n{'=' * 60}")
    result = subprocess.run([sys.executable, step])
    if result.returncode != 0:
        print(f"\n[!] {step} failed. Stopping.")
        sys.exit(result.returncode)

print("\nAll done. Report written to Loan_Default_Prediction_Report.pdf")
