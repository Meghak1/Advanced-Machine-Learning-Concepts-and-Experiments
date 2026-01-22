import matplotlib.pyplot as plt
from data_generation import generate_linear_data
def plot_2d_data(X, y, title = "Linear data"): #plotting the data to visualise
    plt.scatter(X[:,0], X[:,1], c = y)
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("y1")
    plt.show()