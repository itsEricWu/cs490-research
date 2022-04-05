import pickle
import numpy as np
from preprocess import Preprocess
V_list, condition_list = Preprocess.generate_v_matrix(7, 3)
print(condition_list)
print(len(V_list))
pickle.dump(condition_list, open("generated/condition_list", "wb"))
pickle.dump(V_list, open("generated/V_list", "wb"))
