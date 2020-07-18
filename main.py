import csv
import numpy as np
import pandas as pd
from gradinet import gradient_descent

reader = csv.reader(open("datasets_1815_3139_housing.csv", "r"), delimiter=",")
x = list(reader)
result = np.array(x).astype("float")
X = result[:, :-1]
y = result[:, -1:]
theta = np.ones([1, X.shape[1]], dtype=float)

theta = gradient_descent(theta, X, y, 10000, 0.00000016357)
results = np.concatenate((X.dot(np.transpose(theta)), y), axis=1)

pd.DataFrame(results).to_csv("compare2.csv")
