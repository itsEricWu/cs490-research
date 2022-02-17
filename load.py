import pickle
import pandas as pd
import numpy as np

# print("This is the result dataframe")
# result = pickle.load(open("result", "rb"))
# print(result)
# print("This is the analysis dataframe")
# analysis = pickle.load(open("generated/analysis", "rb"))
# print(analysis[(analysis["classfication"] == 10) & (analysis["alpha"] == 0)])

# result = pickle.load(open("generated/changed_v_result", "rb"))
# print(result["predicted label for processed"])
# print(result[(result["alpha"] == 0.2) & (result["condition number"] == -1)])
# print(result[(result["actual label for processed"] == result["predicted label for processed"])
#       & (result["alpha"] == 0.2) & (result["condition number"] == -1)])

V_list = pickle.load(open("generated/changed_v_result", "rb"))
print(V_list)
