#!/usr/bin/env python3
"""
rescore.py — Re-score an existing results.xlsx using the current scorer logic.

Reads the "Results" sheet, recomputes Access Score for each row using
scorer.py, and writes a new sheet "Rescored" with updated scores.
The original sheet is untouched.

Usage:
    python rescore.py                          # uses output/results.xlsx
    python rescore.py path/to/results.xlsx      # custom path
"""
import sys
import pandas as pd
from scorer import compute_access_score

# Column name → internal param key mapping
COL_TO_KEY = {
    "Age": "age",
    "Step Therapy Requirements Documented in Policy": "step_therapy_requirements",
    "Number of Steps through Brands": "steps_through_brands",
    "Number of Steps through Generic": "steps_through_generic",
    "Step through-Phototherapy": "step_through_phototherapy",
    "TB Test required": "tb_test_required",
    "Quantity Limits": "quantity_limits",
    "Specialist Types": "specialist_types",
    "Initial Authorization Duration(in-months)": "initial_auth_duration",
    "Reauthorization Duration(in-months)": "reauth_duration",
    "Reauthorization Required": "reauth_required",
    "Reauthorization Requirements Documented in Policy": "reauth_requirements",
}


def rescore(path: str):
    df = pd.read_excel(path, sheet_name=0)
    print(f"Read {len(df)} rows from {path}")

    old_scores = df["Access Score"].tolist()
    new_scores = []

    for _, row in df.iterrows():
        params = {}
        for col, key in COL_TO_KEY.items():
            val = row.get(col, "NA")
            params[key] = str(val).strip() if pd.notna(val) else "NA"

        brand = str(row.get("Brand", "")).strip()
        score = compute_access_score(params, brand)
        new_scores.append(score)

    df["Access Score"] = new_scores

    # Write both sheets
    with pd.ExcelWriter(path, engine="openpyxl", mode="a",
                        if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name="Rescored", index=False)

    # Summary
    changed = sum(1 for o, n in zip(old_scores, new_scores) if o != n)
    print(f"Rescored: {changed}/{len(df)} rows changed")
    for i, (o, n) in enumerate(zip(old_scores, new_scores)):
        if o != n:
            fname = df.iloc[i]["Filename"]
            brand = df.iloc[i]["Brand"]
            print(f"  {fname} / {brand}: {o} -> {n}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "output/results.xlsx"
    rescore(path)
