import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures

from data_generation import generate_linear_data, generate_xor_data


# create outputs folder
os.makedirs("outputs", exist_ok=True)


def save_plot(X, y, title, filename):
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    plt.savefig(f"outputs/{filename}")
    plt.close()


def main():

    # -------- Linear Data --------
    X_linear, y_linear = generate_linear_data()
    save_plot(X_linear, y_linear, "Linear Data", "linear_data.png")

    # -------- XOR Data --------
    X_xor, y_xor = generate_xor_data()
    save_plot(X_xor, y_xor, "XOR Data", "xor_data.png")

    # -------- Logistic Regression (Linear) --------
    lr = LogisticRegression()
    lr.fit(X_xor, y_xor)
    y_pred_lr = lr.predict(X_xor)

    print("Logistic Regression Accuracy (XOR):",
          accuracy_score(y_xor, y_pred_lr))

    save_plot(
        X_xor,
        y_pred_lr,
        "Logistic Regression on XOR",
        "logistic_linear_xor.png"
    )

    # -------- Polynomial Logistic Regression --------
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_xor)

    lr_poly = LogisticRegression()
    lr_poly.fit(X_poly, y_xor)
    y_pred_poly = lr_poly.predict(X_poly)

    print("Polynomial Logistic Regression Accuracy:",
          accuracy_score(y_xor, y_pred_poly))

    save_plot(
        X_xor,
        y_pred_poly,
        "Polynomial Logistic Regression on XOR",
        "logistic_poly_xor.png"
    )

    # -------- SVM with RBF Kernel --------
    svm = SVC(kernel="rbf")
    svm.fit(X_xor, y_xor)
    y_pred_svm = svm.predict(X_xor)

    print("SVM RBF Accuracy:", accuracy_score(y_xor, y_pred_svm))

    save_plot(
        X_xor,
        y_pred_svm,
        "SVM RBF on XOR",
        "svm_rbf_xor.png"
    )


if __name__ == "__main__":
    main()

