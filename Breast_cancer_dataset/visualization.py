import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os


def plot_2d_data(X, y, title, filename):
    os.makedirs("outputs", exist_ok=True)

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    plt.figure()
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(title)
    plt.savefig(f"outputs/{filename}")
    plt.show()
