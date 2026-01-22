import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score
from data_generation import generate_linear_data
from data_generation import generate_xor_data
from visualization import plot_2d_data
from sklearn.svm import SVC


def main():
    X, y = generate_xor_data(n = 200)
    plot_2d_data(X, y, title = "XOR data (original space)")

#linear SVM

    linear_svm = SVC(kernel='linear')
    linear_svm.fit(X, y)
    y_pred_linear = linear_svm.predict(X)
    plot_2d_data(X, y_pred_linear, title = "Linear SVM Predictions (Fails)")

#polynomial Kernel SVM

    poly_svm = SVC(kernel = 'poly', degree = 2)
    poly_svm.fit(X, y)
    y_pred_poly = poly_svm.predict(X)
    plot_2d_data(X, y_pred_poly, title = "Polynomial Kernel SVM Prediction (Succeeds)")

#RBF SVM Kernel

    rbf_svm = SVC(kernel = "rbf", gamma='scale')
    rbf_svm.fit(X, y)
    y_pred_rbf = rbf_svm.predict(X)
    plot_2d_data(X, y_pred_rbf, title = "RBF Kernel SVM Prediction (Succeeds)")

#print accuracies

    from sklearn.metrics import accuracy_score
    print(f"Linear SVM Accuracy : ", accuracy_score(y, y_pred_linear))
    print(f"Polynomial Kernel SVM Accuracy : ", accuracy_score(y, y_pred_poly))
    print(f"RBF Kernel SVM Accuracy : ", accuracy_score(y, y_pred_rbf))


if __name__ == "__main__":
    main()