from data_generation import generate_linear_data
from visualization import plot_2d_data

def main(): #calling the required functions
    X, y = generate_linear_data()
    plot_2d_data(X, y, title="Linearly Seperable data")

if __name__ == "__main__":
    main()