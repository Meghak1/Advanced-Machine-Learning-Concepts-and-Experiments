from sklearn.svm import SVC


def linear_svm(X, y, C):
    model = SVC(kernel="linear", C=C)
    model.fit(X, y)
    return model.predict(X)


def polynomial_svm(X, y, degree, C):
    model = SVC(kernel="poly", degree=degree, C=C)
    model.fit(X, y)
    return model.predict(X)
