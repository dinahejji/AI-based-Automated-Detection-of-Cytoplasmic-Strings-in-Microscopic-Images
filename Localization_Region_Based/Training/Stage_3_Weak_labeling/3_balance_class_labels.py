# --------------------------------------------
# Balance class labels by undersampling/oversampling
# Author: Mauro Mendez Mora
# --------------------------------------------
# Outputs:
#   • CSV with balanced classification labels
# --------------------------------------------

import pandas as pd

df = pd.read_csv(f"PATH TO CLASSIFICATION LABELS CSV FILE (OUTPUT OF STAGE 3/STEP 2).csv")

# Assume df is your dataframe
df = df.copy()

# Count samples per class
counts = df["Class_GT"].value_counts()
print("Original class distribution:")
print(counts)

# Find minority and majority
min_count = counts.min()
maj_count = counts.max()

# --- Option 1: Undersample majority class (recommended for simplicity) ---
df_balanced_under = (
    df.groupby("Class_GT", group_keys=False)
      .apply(lambda x: x.sample(n=min_count, random_state=42))
)
df_final = df_balanced_under

# # --- Option 2 (alternative): Oversample minority class ---
# df_balanced_over = (
#     df.groupby("Class_GT", group_keys=False)
#       .apply(lambda x: x.sample(n=maj_count, replace=True, random_state=42))
# )
# df_final = df_balanced_over

df_final["Class_GT"] = df_final["Class_GT"].astype(int)

# --- Format indices as 3-digit strings ---
df_final["b0_idx"] = df_final["b0_idx"].apply(lambda x: f"{int(x):03d}")
df_final["b1_idx"] = df_final["b1_idx"].apply(lambda x: f"{int(x):03d}")

# --- Keep only required columns ---
df_final = df_final[["IMG_Name", "b0_idx", "b1_idx", "IoU", "Class_GT"]]

print("Final dataframe preview:")
print(df_final.head())

df_final.to_csv(f"FILE NAME FOR CSV OUTPUT.csv", index=False)