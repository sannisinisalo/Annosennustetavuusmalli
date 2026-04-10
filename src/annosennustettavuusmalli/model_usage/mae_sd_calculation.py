# -*- coding: utf-8 -*-
"""
Luotu Pe 10.04.2026
Tekijä: Sanni Sinisalo

Koodi MAE:n ja SD:n laskemiseen metriikoista
"""

import pandas as pd
import numpy as np

# Lue data
df = pd.read_csv("predicted_doses/dose_metrics.csv")

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
result_df.to_csv(
    "mae_sd_results.csv",
    index=False
)

print("Results saved.")
print(base, len(abs_error))
