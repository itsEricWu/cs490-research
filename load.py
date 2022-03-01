import pickle
import pandas as pd
import numpy as np
import sys


# pixel_mean_variance = pickle.load(open("generated/image_description_results/pixel_mean_variance", "rb"))
# print(pixel_mean_variance)
def out():
    """the first argument from the command line is the file path
    """
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    out = pickle.load(open(sys.argv[1], "rb"))
    print(out.describe())


out()
