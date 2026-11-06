
# ==================================================
# CNC MACHINING QUALITY CLASSIFICATION
# ALGORITHM: DECISION TREE
# ==================================================

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# STEP 1: Locate project folder
project_folder = Path(__file__).resolve().parent.parent

# STEP 2: Load original dataset
file_path = project_folder / "data" / "cnc_surface_roughness_data.csv"
data = pd.read_csv(file_path)

# STEP 3: Remove rows with missing Ra
data = data.dropna(subset=["Ra"]).copy()

# STEP 4: Select input variables
features = ["fz", "ap", "Vc", "vb", "C"]
X = data[features]

# STEP 5: Create relative quality categories
data["Quality"] = pd.qcut(
    data["Ra"],
    q=3,
    labels=["Low", "Medium", "High"]
)

y = data["Quality"]

print("QUALITY CATEGORY DISTRIBUTION")
print(y.value_counts())

# STEP 6: Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# STEP 7: Create Decision Tree model
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    min_samples_leaf=2,
    random_state=42
)

# STEP 8: Train the model
model.fit(X_train, y_train)

# STEP 9: Make predictions
y_pred = model.predict(X_test)

# STEP 10: Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("\nDECISION TREE RESULTS")
print("---------------------")
print("Accuracy:", round(accuracy, 4))

print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

# STEP 11: Confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["Low", "Medium", "High"]
)

print("\nCONFUSION MATRIX")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Low", "Medium", "High"]
)

display.plot(cmap="Blues")
plt.title("Decision Tree Confusion Matrix")
plt.savefig(
    project_folder / "outputs" / "decision_tree_confusion_matrix.png"
)
plt.show()

# STEP 12: Visualize Decision Tree
plt.figure(figsize=(16, 9))

plot_tree(
    model,
    feature_names=features,
    class_names=["Low", "Medium", "High"],
    filled=True,
    rounded=True,
    fontsize=9
)

plt.title("Decision Tree for CNC Surface Quality")
plt.savefig(
    project_folder / "outputs" / "decision_tree_structure.png"
)
plt.show()

# STEP 13: Feature importance
importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFEATURE IMPORTANCE")
print(importance)

# STEP 14: Save classification results
results = X_test.copy()
results["Actual Quality"] = y_test
results["Predicted Quality"] = y_pred

output_path = (
    project_folder / "outputs" / "decision_tree_results.csv"
)

results.to_csv(output_path, index=False)

print("\nDECISION TREE COMPLETED!")
print("Results saved at:", output_path)
