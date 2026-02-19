import matplotlib.pyplot as plt
from generation_of_twolakhdatapoints import generate_linear_data

def plot_and_save_large_data(X, y, title = "Linear data"): #plotting the data to visualise
    plt.scatter(X[:,0], X[:,1], c = y)
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("y1")
    plt.show()