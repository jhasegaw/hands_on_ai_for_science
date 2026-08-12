"""
Generate the deliberately-messy example dataset for the Aug 12 afternoon
session "Your Data as a Matrix" (a12pm_modules.ipynb).

Attendees do not need to run this file -- the CSVs it produces are already
in the repository.  It is kept here so the dataset's provenance is clear
and so the mess is reproducible (fixed random seed).

Produces:
  a12pm_measurements.csv    - 300 rows: 5 samples from each of 60 mice.
                              Deliberate problems: inconsistent capitalization
                              in `condition`, scattered missing values in
                              gene_a..gene_c, and gene_d stored as text
                              because a few entries say "n.d." (not detected).
  a12pm_mouse_metadata.csv  - 60 rows: one per mouse (sex, age, batch).
"""

import numpy as np
import pandas as pd

N_MICE = 60
SAMPLES_PER_MOUSE = 5

def main():
    rng = np.random.default_rng(2026)

    mouse_ids = [f"m{i+1:03d}" for i in range(N_MICE)]
    condition = np.array(["control"] * (N_MICE // 2) + ["treatment"] * (N_MICE // 2))
    sex = rng.choice(["F", "M"], N_MICE)
    age_weeks = rng.integers(8, 30, N_MICE)
    batch = np.repeat(["batch1", "batch2", "batch3"], N_MICE // 3)

    meta = pd.DataFrame({
        "mouse_id": mouse_ids,
        "sex": sex,
        "age_weeks": age_weeks,
        "batch": batch,
    })

    rows = []
    variants = {
        "control": ["control", "Control", "CONTROL", "control "],
        "treatment": ["treatment", "Treatment", "TREATMENT", "treatment "],
    }
    sample_n = 0
    for m in range(N_MICE):
        treat = 1.0 if condition[m] == "treatment" else 0.0
        mouse_effect = rng.normal(0, 0.3)
        for s in range(SAMPLES_PER_MOUSE):
            sample_n += 1
            rows.append({
                "sample_id": f"s{sample_n:03d}",
                "mouse_id": mouse_ids[m],
                # the condition column is entered by hand, so capitalization drifts:
                "condition": variants[condition[m]][rng.integers(0, 4)],
                "gene_a": round(rng.normal(5, 1) + 1.2 * treat + mouse_effect, 3),
                "gene_b": round(rng.normal(3, 1) + mouse_effect, 3),
                "gene_c": round(rng.normal(10, 2) + 0.08 * age_weeks[m] + 0.3 * treat, 3),
                "gene_d": round(rng.normal(7, 1.5) + 0.5 * treat, 3),
            })
    df = pd.DataFrame(rows)

    # scatter missing values through gene_a..gene_c (~5% of cells)
    for col in ["gene_a", "gene_b", "gene_c"]:
        miss = rng.choice(len(df), size=15, replace=False)
        df.loc[miss, col] = np.nan

    # a few gene_d entries were below the detection threshold, and the lab
    # notebook says "n.d." -- which turns the whole column into text
    df["gene_d"] = df["gene_d"].astype(object)
    nd = rng.choice(len(df), size=8, replace=False)
    df.loc[nd, "gene_d"] = "n.d."

    df.to_csv("a12pm_measurements.csv", index=False)
    meta.to_csv("a12pm_mouse_metadata.csv", index=False)
    print(f"wrote a12pm_measurements.csv ({len(df)} rows) and "
          f"a12pm_mouse_metadata.csv ({len(meta)} rows)")

if __name__ == "__main__":
    main()
