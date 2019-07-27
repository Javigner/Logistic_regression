import numpy as np
import pandas as pd

def sigmoid(z):
    return (1 / (1 + np.exp(-z)))
   
def predict(X, theta):
      return(sigmoid(np.dot(X, theta)))
      
def standar_scaler(X):
    mean = X.mean(axis=0)
    stdev = X.std(axis=0)
    X = (X - mean)/stdev
    return X, mean, stdev

def gradient_descent(theta, X, y):
    learning_rate = 0.0001
    X, mean, stdev = standar_scaler(X)
    X = np.c_[np.ones(X.shape[0]), X]
    testha = [1.0, 1.0, 1.0]
    testha = np.array(testha)
    learning_rate = 0.1
    print(theta[0], theta[1], theta[2])
    while (testha.all() != theta.all()):
        theta[0] = testha[0]
        theta[1] = testha[1]
        theta[2] = testha[2]
        testha[0] = theta[0] - learning_rate * (1.0 / len(X)) * (np.dot(predict(X, theta) - y, X))
        testha[1] = theta[1] - learning_rate * (1.0 / len(X)) * (np.dot(predict(X, theta) - y, X))
        testha[2] = theta[2] - learning_rate * (1.0 / len(X)) * (np.dot(predict(X, theta) - y, X))
    return theta, mean, stdev
    

def main():
    data = pd.read_csv('ex2data1.csv')
    X = data.values[:, 0:2]
    y = data.values[:, -1]
    X = np.array(X)
    y = np.array(y)
    theta = np.zeros(3)
    theta, mean, stdev = gradient_descent(theta, X, y)
    X_test = (np.array([34,78]) - mean) / stdev
    X_test = np.hstack([1, X_test])
    print(predict(X_test, theta))
    

if __name__ == "__main__":
    main();


