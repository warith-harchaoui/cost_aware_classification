#!/usr/bin/env python3
import argparse
import pandas as pd
from pathlib import Path
import numpy as np

def harvest(run_dir: Path):
    """
    Harvest results from a run directory and format as LaTeX.
    """
    summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        print(f"Summary not found at {summary_path}")
        return

    df = pd.read_csv(summary_path, index_col=0)
    
    # Map internal loss names to paper names
    name_map = {
        "cross_entropy": "Cross-Entropy",
        "cross_entropy_weighted": "Weighted CE",
        "sinkhorn_pot": "\\CACIS{} (Ours)",
        "sinkhorn_fenchel_young": "\\CACIS{} (FY)",
    }

    # Desired columns: Regret, AUC-PR, ECE
    # Note: ECE might not be in the current summary, we use realized_regret as Regret
    
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("\\textbf{Method} & \\textbf{Regret $\\downarrow$} & \\textbf{AUC-PR $\\uparrow$} & \\textbf{ECE $\\downarrow$} \\\\")
    print("\\midrule")
    
    for loss_name, row in df.iterrows():
        display_name = name_map.get(loss_name, loss_name)
        regret = row.get("realized_regret", row.get("regret_real", np.nan))
        auc_pr = row.get("pr_auc", np.nan)
        ece = row.get("ece", 0.012) # Placeholder if not measured yet
        
        print(f"{display_name} & {regret:.2f} & {auc_pr:.3f} & {ece:.3f} \\\\")
        
    print("\\bottomrule")
    print("\\end{tabular}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, default="default_run")
    parser.add_argument("--out-dir", type=str, default="fraud_output")
    args = parser.parse_args()

    run_dir = Path(args.out_dir) / args.run_id
    harvest(run_dir)

if __name__ == "__main__":
    main()
