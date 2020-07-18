import numpy as np


def cost_function(theta, X, y):
    return (1/2/X.shape[1]) * np.sum( np.square(X.dot(np.transpose(theta)) - y))


def gradient_descent(theta, X, y, itr, alpha):
    it = 1
    print(theta.shape)
    print(X.shape)
    print(y.shape)

    while it <= itr:
        hypothesis = X.dot(np.transpose(theta)) - y
        print(X.dot(np.transpose(theta)).shape)
        res = (np.transpose(X)).dot(hypothesis)
        sub = (alpha/X.shape[1]) * np.transpose(res)
        print("I want to subtract")
        print(sub)
        theta = theta - sub
        print("print cost after {0}th iteration : {1}".format(it, cost_function(theta, X, y)))
        print("Result of theta after " + str(it) + "th iteration")
        print(theta)
        it += 1
    return theta
