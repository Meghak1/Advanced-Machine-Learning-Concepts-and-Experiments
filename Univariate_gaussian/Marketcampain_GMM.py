import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

# Load dataset
data = pd.read_csv("C:/Advanced Machine Learning Concepts and Experiments/Univariate_gaussian/marketing_campaign.csv", sep=";")

# Drop unnecessary columns
data = data.drop(columns=["ID", "Dt_Customer"])

# Handle missing values (Income)
data["Income"].fillna(data["Income"].median(), inplace=True)

# Encode categorical variables
categorical_cols = ["Education", "Marital_Status"]
le = LabelEncoder()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

# Scale features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Apply Gaussian Mixture Model with 2 components
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(scaled_data)
gmm_clusters = gmm.predict(scaled_data)

# Add cluster labels to dataframe
data["GMM_Cluster"] = gmm_clusters

# PCA for 2D visualization
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# Plot GMM clusters
plt.figure(figsize=(8,6))
plt.scatter(pca_data[:,0], pca_data[:,1],
            c=gmm_clusters, cmap='coolwarm', s=50)
plt.title("Gaussian Mixture Model Clustering (2 Components)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster")
plt.grid(True)
plt.show()
