import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Breast cancer imports
from load_breast_cancer import load_data
from EDA import perform_eda
from svm_models import linear_svm, polynomial_svm
from visualization import plot_2d_data

# Large data imports
from generation_of_twolakhdatapoints import (
    generate_linear_data,
    generate_non_linear_data
)
from visualisation_linear_data import plot_and_save_large_data


def subsample(X, y, n_samples=2000):
    """
    Subsample large dataset for visualization
    """
    idx = np.random.choice(len(X), n_samples, replace=False)
    return X[idx], y[idx]


def main():
    os.makedirs("outputs", exist_ok=True)

    print("\n========== BREAST CANCER DATASET ==========\n")

    X, y = load_data()

    perform_eda(X, y)

    X_np = X.values
    y_np = y.values

    # Linear Soft Margin
    y_pred_linear_soft = linear_svm(X_np, y_np, C=2)
    print("Linear SVM (Soft Margin) Accuracy:",
          accuracy_score(y_np, y_pred_linear_soft))

    plot_2d_data(
        X_np,
        y_pred_linear_soft,
        "Linear SVM (Soft Margin, C=2)",
        "linear_svm_soft_margin.png"
    )

    # Linear Hard Margin
    y_pred_linear_hard = linear_svm(X_np, y_np, C=1000)
    print("Linear SVM (Hard Margin) Accuracy:",
          accuracy_score(y_np, y_pred_linear_hard))

    plot_2d_data(
        X_np,
        y_pred_linear_hard,
        "Linear SVM (Hard Margin, C=1000)",
        "linear_svm_hard_margin.png"
    )

    # Polynomial SVM
    y_pred_poly_bc = polynomial_svm(X_np, y_np, degree=2, C=2)
    print("Polynomial Kernel SVM Accuracy:",
          accuracy_score(y_np, y_pred_poly_bc))

    plot_2d_data(
        X_np,
        y_pred_poly_bc,
        "Polynomial Kernel SVM (Degree=2)",
        "polynomial_kernel_svm_bc.png"
    )

    # ===============================
    # LARGE SYNTHETIC DATA
    # ===============================

    print("\n LARGE SYNTHETIC DATA\n")

    # Linear Data
    X_lin, y_lin = generate_linear_data(n=200000)

    svm_linear = SVC(kernel="linear")
    svm_linear.fit(X_lin, y_lin)
    y_pred_lin = svm_linear.predict(X_lin)

    print("Linear SVM Accuracy on Linear Data:",
          accuracy_score(y_lin, y_pred_lin))

    X_vis, y_vis = subsample(X_lin, y_pred_lin)
    plot_and_save_large_data(X_vis, y_vis,
                             title="Linear SVM on Linear Data")

    # Non-Linear Data
    X_nonlin, y_nonlin = generate_non_linear_data(n=200000)

    svm_linear.fit(X_nonlin, y_nonlin)
    y_pred_linear = svm_linear.predict(X_nonlin)

    print("Linear SVM Accuracy on Non-Linear Data:",
          accuracy_score(y_nonlin, y_pred_linear))

    X_vis, y_vis = subsample(X_nonlin, y_pred_linear)
    plot_and_save_large_data(X_vis, y_vis,
                             title="Linear SVM on Non-Linear Data (Fails)")

    # Polynomial Kernel
    svm_poly = SVC(kernel="poly", degree=2)
    svm_poly.fit(X_nonlin, y_nonlin)
    y_pred_poly = svm_poly.predict(X_nonlin)

    print("Polynomial Kernel SVM Accuracy:",
          accuracy_score(y_nonlin, y_pred_poly))

    X_vis, y_vis = subsample(X_nonlin, y_pred_poly)
    plot_and_save_large_data(X_vis, y_vis,
                             title="Polynomial Kernel SVM on Non-Linear Data")

    # RBF Kernel
    svm_rbf = SVC(kernel="rbf", gamma="scale")
    svm_rbf.fit(X_nonlin, y_nonlin)
    y_pred_rbf = svm_rbf.predict(X_nonlin)

    print("RBF Kernel SVM Accuracy:",
          accuracy_score(y_nonlin, y_pred_rbf))

    X_vis, y_vis = subsample(X_nonlin, y_pred_rbf)
    plot_and_save_large_data(X_vis, y_vis,
                             title="RBF Kernel SVM on Non-Linear Data")


if __name__ == "__main__":
    main()
