
# ==========================================
# CNC MACHINING DATA PREPROCESSING
# ==========================================

import pandas as pd
from pathlib import Path

# Locate project folder
project_folder = Path(__file__).resolve().parent.parent

# Load original dataset
file_path = project_folder / "data" / "cnc_surface_roughness_data.csv"
data = pd.read_csv(file_path)

print("Original dataset shape:", data.shape)

# Select input parameters
features = ["fz", "ap", "Vc", "vb", "C"]

# Remove rows where target Ra is missing
clean_data = data.dropna(subset=["Ra"]).copy()

# Separate input variables (X)
X = clean_data[features]

# Separate output variable (y)
y = clean_data["Ra"]

# Save cleaned dataset
output_path = project_folder / "outputs" / "cleaned_cnc_data.csv"
clean_data.to_csv(output_path, index=False)

# Display results
print("\nPREPROCESSING COMPLETED")
print("Cleaned dataset shape:", clean_data.shape)
print("\nInput variables:")
print(X.head())

print("\nTarget variable (Ra):")
print(y.head())

print("\nMissing values after cleaning:")
print(clean_data.isnull().sum())

print("\nCleaned dataset saved at:", output_path)
