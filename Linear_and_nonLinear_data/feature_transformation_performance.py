import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score
from data_generation import generate_linear_data
from data_generation import generate_xor_data
from visualization import plot_2d_data

#generate xor data

X,y=generate_xor_data(n=200)

plot_2d_data(X, y, title="XOR data (original space)")

#linear model without feature transformation

linear_model = LogisticRegression()
linear_model.fit(X, y)
y_pred_linear = linear_model.predict(X)
linear_accuracy=accuracy_score(y, y_pred_linear)
print(f"Linear model acuuracy on XOR data : {linear_accuracy:.2f}")
plot_2d_data(X, y_pred_linear, title = "XOR data - Linear model prediction")

# this gives wrong prediction as the data is not non linear and we are applying linear model on non linear data

#feature transformation to polynomial features

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
#linear model on transformed features
poly_model = LogisticRegression()
poly_model.fit(X_poly, y)
y_pred_poly = poly_model.predict(X_poly)
poly_accuracy = accuracy_score(y, y_pred_poly) 
print(f"Polynomial feature model accuracy on xor data : {poly_accuracy:.2f}")
plot_2d_data(X, y_pred_poly, title = "XOR data - Polynomial Feature Model Prediction")

