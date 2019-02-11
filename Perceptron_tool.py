"""
CMPE 452 Assignment 1
Curtis Shewchuk
14cms13
SN: 10189026
"""
from sklearn import linear_model
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np

def load_data(file):
    """
    Parses a file to be used in sklearn
    :param file name of the file:
    :return X: inputs:
    :return Y: outputs:
    """

    X,Y = [],[]
    for line in open(file).readlines():
        strip = line.rstrip().split(',')
        X.append([float(v) for v in strip[:-1]])
        if strip[-1] == '1':
            Y.append([1, 0, 0])
        elif strip[-1] == '2':
            Y.append([0, 1, 0])
        elif strip[-1] == '3':
            Y.append([0, 0, 1])

    X = np.matrix(X)
    Y = np.matrix(Y)
    return X,Y

X_train, d_train = load_data("trainSeeds.csv")
X_test, d_test = load_data("testSeeds.csv")
for i in range(3):
    p = linear_model.Perceptron()
    dn = np.ravel(d_train[:, i])
    p.fit(X_train, dn)
    y = p.predict(X_test)
    precision = precision_score(np.ravel(d_test[:, i]), y)
    recall = recall_score(np.ravel(d_test[:, i]), y)
    print("Output", i + 1, precision * 100, recall * 100)