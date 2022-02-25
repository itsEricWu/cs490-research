import numpy as np
import cv2
from matplotlib import pyplot as plt
import pickle
from preprocess import Preprocess


v, c = Preprocess.generate_constant_v_matrix(10, 1, True)
epsilons = [0, 0.05, 0.1, 0.15, 0.2]
for i, k in zip(v, c):
    for j in epsilons:
        x_new = Preprocess.preprocess_image("Test/0/10.ppm", i, j)
        filename = "generated/jpgimages/10_" + str(k) + "_" + str(j)
        Preprocess.show_np_array_as_jpg(x_new, filename)
