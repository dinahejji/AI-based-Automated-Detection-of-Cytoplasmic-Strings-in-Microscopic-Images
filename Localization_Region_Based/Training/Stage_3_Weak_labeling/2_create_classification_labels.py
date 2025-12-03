# --------------------------------------------
# Create weakly classification labels based on IoU thresholds
# Higher than 0.2 -> Positive (1)
# Lower than 0.1  -> Negative (0)
# Author: Mauro Mendez Mora
# --------------------------------------------
# Outputs:
#   • CSV with classification labels
# --------------------------------------------

import pandas as pd

df = pd.read_csv(f"PATH TO IOU CSV FILE (OUTPUT OF STAGE 3/STEP 1).csv")

# ensure numeric IoU and drop NaN
df = df.copy()
df["IoU"] = pd.to_numeric(df["IoU"], errors="coerce")
df = df.dropna(subset=["IoU"])

# classify based on IoU thresholds
df["Class_GT"] = df["IoU"].apply(
    lambda x: 1 if x > 0.2 else (0 if x < 0.1 else None)
)

# drop ignored rows (IoU between 0.1 and 0.2)
df_final = df.dropna(subset=["Class_GT"]).copy()
df_final["Class_GT"] = df_final["Class_GT"].astype(int)

# --- Format indices as 3-digit strings ---
df_final["b0_idx"] = df_final["b0_idx"].apply(lambda x: f"{int(x):03d}")
df_final["b1_idx"] = df_final["b1_idx"].apply(lambda x: f"{int(x):03d}")

# --- Keep only required columns ---
df_final = df_final[["IMG_Name", "b0_idx", "b1_idx", "IoU", "Class_GT"]]

print("Final dataframe preview:")
print(df_final.head())

df_final.to_csv(f"FILE NAME FOR CSV OUTPUT.csv", index=False)