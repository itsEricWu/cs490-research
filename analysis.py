import pickle
import pandas as pd
import numpy as np

df = pickle.load(open("result", "rb"))

original_count = 0
correct_count = 0
for index, row in df.iterrows():
    if row["alpha"] == 0.8 and not np.array_equal(np.array(row["v"]), np.identity(3)):
        original_count += 1
        if row["actual label for processed"] == row["predicted label for processed"]:
            correct_count += 1
accu = correct_count / original_count
print(correct_count)
print(original_count)
print(accu)
