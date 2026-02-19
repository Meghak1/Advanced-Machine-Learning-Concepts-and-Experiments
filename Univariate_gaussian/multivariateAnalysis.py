import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

np.random.seed(42)

#generate data

daily = np.random.normal(5, 0.5, 100)
peak = np.random.normal(2, 0.3, 100)

X = np.column_stack((daily, peak))

#estimate parameters

mean_vector = np.mean(X, axis=0)
cov_matrix = np.cov(X.T)

print("mean vector : \n", mean_vector)
print("Covariance matrix : \n",cov_matrix)

#create gaussian model
rv = multivariate_normal(mean_vector, cov_matrix)

#create grid
x, y = np.mgrid[3:7:0.05, 1:3:0.05]
pos = np.dstack((x, y))

#plot countor
plt.figure(figsize=(6,5))
plt.contour(x, y, rv.pdf(pos), levels=20, cmap='viridis')
plt.colorbar(label='Probability Density')

plt.scatter(X[:, 0], X[:, 1], c='red', edgecolors='k', label='Data Points')

plt.title("Multivariate Gaussian Distribution")
plt.xlabel("Daily Usage")
plt.ylabel("Peak Usage")
plt.legend()
plt.show()

