import numpy as np
import cv2
from matplotlib import pyplot as plt
import pickle

def generate_v_matrix(num_condition=10, num_matrix=10, identity=True):
    np.random.seed(0)
    a = np.random.normal(0, 1, size=(3, 3))  # random matrix
    print(a)
    eps_list = np.logspace(0, 3, num_condition)
    V_list = []
    condition_list = []
    for eps in eps_list:
        for i in range(0, num_matrix):
            C = eps
            u, s, v = np.linalg.svd(a, full_matrices=True) #svd
            s = s[0] * (1 - ((C - 1) / C) * (s[0] - s) / (s[0] - s[-1])) #linear stretch
            s = np.diag(s)
            V = u @ s @ v
            V_list.append(V)
            condition_list.append(eps)
    if identity:
        V_list.append(np.identity(3))
        condition_list.append(-1)
    return V_list, condition_list

v, c = generate_v_matrix(10, 1, True)
print(v)
