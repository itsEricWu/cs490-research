import pickle
import pandas as pd

df = pickle.load(open("result", "rb"))
# print(df[df["actual label for processed"] != df["predicted label for processed"]])
print(df)
