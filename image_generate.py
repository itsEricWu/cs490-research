import numpy as np
import cv2
from matplotlib import pyplot as plt
import pickle
from preprocess import Preprocess


v, c = Preprocess.generate_constant_v_matrix(7, 3, True)
epsilons = [0, 0.05, 0.1, 0.15, 0.2]
for i, k in zip(v, c):
    for j in epsilons:
        x_new = Preprocess.preprocess_image_gaussian("Test/0/32.ppm", i, j)
        filename = "generated/jpgimages/10_" + str(k) + "_" + str(j) + "_gaussian"
        Preprocess.show_np_array_as_jpg(x_new, filename)
        x_new = Preprocess.preprocess_image_speckle("Test/0/32.ppm", i, j)
        filename = "generated/jpgimages/10_" + str(k) + "_" + str(j) + "_speckle"
        Preprocess.show_np_array_as_jpg(x_new, filename)
        x_new = Preprocess.preprocess_image_poisson("Test/0/32.ppm", i, j)
        filename = "generated/jpgimages/10_" + str(k) + "_" + str(j) + "_poisson"
        Preprocess.show_np_array_as_jpg(x_new, filename)
