from data_generation import generate_xor_data
from visualization import plot_2d_data

def main(): #calling the required functions
    X, y = generate_xor_data()
    plot_2d_data(X, y, title="Non linearly seperable data")

if __name__ == "__main__":
    main()