import pandas as pd
import numpy as np

np_array = np.ones([5, 2], dtype="float")
pd.DataFrame(np_array).to_csv("demo.csv")
