import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler

v = pickle.load(open("generated/V_list", "rb"))
max_v = np.maximum.reduce(v)
min_v = np.minimum.reduce(v)
f_max_v = max_v.flatten()
f_min_v = min_v.flatten()
f_max_v = np.append(f_max_v, np.array([0.2, 1000]))
f_min_v = np.append(f_min_v, np.array([0, -1]))
min_max = np.append([f_max_v], [f_min_v], axis=0)
scaler = MinMaxScaler()
scaler.fit(min_max)
pickle.dump(scaler, open("generated/scaler", "wb"))
