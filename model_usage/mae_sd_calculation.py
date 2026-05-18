# -*- coding: utf-8 -*-
"""
Luotu Pe 10.04.2026
Tekijä: Sanni Sinisalo

Koodi MAE:n ja SD:n laskemiseen metriikoista
"""

import pandas as pd
import numpy as np
from luokat2 import BASE_DIR

# Lue data
path = BASE_DIR / "predicted_doses" / "dose_metrics.csv"
df = pd.read_csv(path)

results = []

for col in df.columns:

    if col.endswith("_pred"):

        base = col.replace("_pred", "")

        pred_col = base + "_pred"
        clin_col = base + "_clin"

        # Absolute error per patient
        abs_error = np.abs(
            df[pred_col] - df[clin_col]
        )

        # Mean absolute error
        mae = abs_error.mean()

        # Standard deviation
        sd = abs_error.std()

        results.append({
            "Metric": base,
            "MAE": mae,
            "SD": sd,
            "Result": f"{mae:.3f} ± {sd:.3f}"
        })

# DataFrame
result_df = pd.DataFrame(results)

# Tallenna
out_path = BASE_DIR / "predicted_doses" / "mae_sd_results.csv"

result_df.to_csv(
    out_path,
    index=False
)

print("Results saved.")
