
# ==================================================
# CNC MACHINING DATA ANALYSIS
# ALGORITHM: K-MEANS CLUSTERING
# ==================================================

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# STEP 1: Locate project folder
project_folder = Path(__file__).resolve().parent.parent

# STEP 2: Load cleaned dataset
file_path = project_folder / "data" / "cnc_surface_roughness_data.csv"
data = pd.read_csv(file_path)

# STEP 3: Select machining parameters
features = ["fz", "ap", "Vc", "vb", "C"]
X = data[features]

# STEP 4: Scale the input data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Original data shape:", X.shape)
print("Scaled data shape:", X_scaled.shape)

# STEP 5: Find suitable number of clusters
inertias = []
silhouette_scores = []
k_values = range(2, 7)

for k in k_values:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    inertias.append(model.inertia_)
    silhouette_scores.append(
        silhouette_score(X_scaled, labels)
    )

# Display cluster evaluation
print("\nCLUSTER EVALUATION")

for k, inertia, score in zip(
    k_values, inertias, silhouette_scores
):
    print(
        "K:", k,
        "| Inertia:", round(inertia, 4),
        "| Silhouette:", round(score, 4)
    )

# STEP 6: Plot elbow graph
plt.figure(figsize=(7, 5))

plt.plot(
    list(k_values),
    inertias,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")

plt.savefig(
    project_folder / "outputs" / "elbow_plot.png"
)

plt.show()

# STEP 7: Train final K-Means model
# We start with K = 3 for demonstration.
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans.fit_predict(X_scaled)

# STEP 8: Add cluster labels to dataset
clustered_data = data.copy()
clustered_data["Cluster"] = cluster_labels

# STEP 9: Display cluster information
print("\nCLUSTER ASSIGNMENTS")
print(
    clustered_data[
        ["Run", "Ra", "Cluster"]
    ].to_string(index=False)
)

print("\nNUMBER OF SAMPLES IN EACH CLUSTER")
print(clustered_data["Cluster"].value_counts().sort_index())

# STEP 10: Calculate cluster-wise averages
print("\nCLUSTER-WISE MACHINING PARAMETERS")
print(
    clustered_data.groupby("Cluster")[features + ["Ra"]]
    .mean()
    .round(3)
)

# STEP 11: Reduce five dimensions to two for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=cluster_labels,
    cmap="viridis",
    s=70
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Clustering of CNC Machining Data")
plt.colorbar(label="Cluster")

plt.savefig(
    project_folder / "outputs" / "kmeans_clusters.png"
)

plt.show()

# STEP 12: Save clustered dataset
output_path = (
    project_folder / "outputs" / "clustered_cnc_data.csv"
)

clustered_data.to_csv(output_path, index=False)

print("\nK-MEANS CLUSTERING COMPLETED!")
print("Results saved at:", output_path)
