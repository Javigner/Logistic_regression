import numpy as np
import pandas as pd

def gradient_descent(theta, X, y):
    test
    

def main():
    data = pd.read_csv('ex2data1.csv')
    X = data.values[:, 0:2]
    y = data.values[:, -1]
    X = np.array(X)
    y = np.array(y)
    theta = np.zeros(2)
    theta = gradient_descent(theta, X, y)
    

if __name__ == "__main__":
    main();


