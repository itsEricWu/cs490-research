import pickle
import numpy as np
from preprocess import Preprocess
import cv2

def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out

def preprocess_image_gaussian(img_filename):
    """
    processs the image with v matrix and normal noise
    input:
    img_filename: the path of an image in ppm format
    v matrix: 3 by 3 numpy array
    alpha: coefficient used for generating gaussian noise
    """
    # Comments on v matrix:
    # For some global color mixing matrix v, figure out what the input to the
    # projector should be, assuming zero illumination other than the projector.
    # Note : Not all scenes are possible for all v matrices. For some matrices
    # some scenes are not possible at all.

    img = cv2.imread(img_filename)  # numpy array
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Captured Image - original #
    print(img)
    y = Preprocess.im2double(img)  # Convert to
    print(y)


preprocess_image_gaussian("/home/lu677/cs490/cs490-research/Test/0/10.ppm")
