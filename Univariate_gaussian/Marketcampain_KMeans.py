import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Create dataframe manually (since raw text provided)
data = pd.read_csv("C:/Advanced Machine Learning Concepts and Experiments/Univariate_gaussian/marketing_campaign.csv", sep=";")

# Drop ID and date column
data = data.drop(columns=["ID", "Dt_Customer"])


# Handle missing values
data["Income"].fillna(data["Income"].median(), inplace=True)


# Encode categorical columns
categorical_cols = ["Education", "Marital_Status"]
le = LabelEncoder()

for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

# Scale features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Apply KMeans with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Add cluster labels
data["Cluster"] = clusters

# PCA for visualization (2D)
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# Plot
plt.figure(figsize=(8,6))
plt.scatter(pca_data[:,0], pca_data[:,1],
            c=clusters, cmap='viridis', s=120)

plt.title("K-Means Clustering (k=2)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster")
plt.grid(True)
plt.show()
