
# ==========================================
# CNC MACHINING DATASET INSPECTION
# ==========================================

# 1. Import libraries
import pandas as pd
from pathlib import Path

# 2. Locate and load dataset
project_folder = Path(__file__).resolve().parent.parent
file_path = project_folder / "data" / "cnc_surface_roughness_data.csv"

data = pd.read_csv(file_path)

# 3. Display first five records
print("\n--- FIRST FIVE RECORDS ---")
print(data.head())

# 4. Display dataset dimensions
print("\n--- DATASET SHAPE ---")
print(data.shape)

# 5. Display column information
print("\n--- DATASET INFORMATION ---")
data.info()

# 6. Check missing values
print("\n--- MISSING VALUES ---")
print(data.isnull().sum())

# 7. Statistical summary
print("\n--- STATISTICAL SUMMARY ---")
print(data.describe())

print("\nDataset inspection completed!")
