import pickle
import pandas as pd
import numpy as np

print("This is the result dataframe")
result = pickle.load(open("result", "rb"))
print(result)
print("This is the analysis dataframe")
analysis = pickle.load(open("analysis", "rb"))
print(analysis)
