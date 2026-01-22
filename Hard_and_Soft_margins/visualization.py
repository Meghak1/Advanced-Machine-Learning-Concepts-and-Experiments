import numpy as np
import matplotlib.pyplot as plt


def plot_data(X, y, title):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
    plt.title(title)
    plt.show()


def plot_svm(X, y, model, title):
    w = model.coef_[0]
    b = model.intercept_[0]

    plt.figure(figsize=(7, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')

    plt.scatter(
        model.support_vectors_[:, 0],
        model.support_vectors_[:, 1],
        s=120, facecolors='none', edgecolors='k', linewidths=2,
        label="Support Vectors"
    )

    x_vals = np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 200)

    y_decision = -(w[0]*x_vals+b)/w[1]
    y_margin_pos = -(w[0]*x_vals+b-1)/w[1]
    y_margin_neg = -(w[0]*x_vals+b+1)/w[1]

    plt.plot(x_vals, y_decision, 'k--', label="Decision boundary")
    plt.plot(x_vals, y_margin_pos, 'k--', label="margin +1")
    plt.plot(x_vals, y_margin_neg, 'k--', label="margin -1")

    plt.fill_between(x_vals, y_margin_pos, y_margin_neg, alpha=0.2, label="Margin Area")

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.legend()
    plt.show()