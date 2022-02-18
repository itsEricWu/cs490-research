from os import listdir
from os.path import isfile, join
import random
import cv2
import numpy as np
from matplotlib import pyplot as plt

selected_files = []
for i in range(19):
    mypath = f"Test/{i}"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    selected_file = random.choice(onlyfiles)
    selected_files.append([i, selected_file])
print(selected_files)

 
for i in range(len(selected_files)):
    img = cv2.imread(f"Test/{selected_files[i][0]}/{selected_files[i][1]}", 0)
    plt.hist(img.ravel(),256,[0,256])
    path = 'pixel_histograms'
    plt.savefig(f'{path}/histogram{i}.jpg')

