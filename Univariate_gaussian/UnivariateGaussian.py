import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

data = np.array([4.1, 4.3, 4.8, 5.0, 5.2, 5.5, 6.0, 6.2, 4.9, 5.1, 5.3, 5.8, 6.1, 4.7, 5.4, 5.6, 4.6, 6.3, 5.9, 5.0])
mean = np.mean(data)
variance = np.var(data)
print("mean : ", mean)
print("Variance : ", variance)

x = np.linspace(min(data)-1, max(data)+1, 100)
pdf = norm.pdf(x, mean, np.sqrt(variance))

plt.hist(data, bins = 8, density=True, alpha = 0.6, label = "Data Histogram")
plt.plot(x, pdf, 'r', label = "Gaussian PDF")
plt.title("UNIVARIATE GAUSSIAN DISTRIBUTION")
plt.xlabel("Electricity usage (kwh)")
plt.ylabel("Density")
plt.legend()
plt.show()

