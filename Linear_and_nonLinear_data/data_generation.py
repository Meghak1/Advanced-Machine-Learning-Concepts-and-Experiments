import numpy as np

def generate_linear_data(n=100): #to generate linear data
    X = np.random.randn(n, 2) #100 rows, two columns
    y = (X[:,0]+X[:,1] > 0).astype(int)  #y 
    return X,y 

# def generate_non_linear_data(n=200):
#     X = np.random.randn(n, 2)
#     y = (X[:, 0]**2 + X[:, 1]**2 > 1).astype(int)
#     return X, y

def generate_xor_data(n=200): # to generate xor data
    X = np.random.randn(n, 2) #200 rows, 2 colums , generates data in format at 0,1 range
    y = ((X[:,0]>0) ^ (X[:,1]>0)).astype(int)
    return X,y