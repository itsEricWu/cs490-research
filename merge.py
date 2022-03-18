from copyreg import pickle
from tqdm import tqdm
import os
import pickle
import pandas as pd
import numpy as np


def merge_result():
    df = pd.DataFrame()
    folder = "./generated/unmerged_result/"
    g = os.walk(folder)
    f = 0
    for path, dir_list, file_list in g:
        for file_name in tqdm(file_list, desc=str(f)):
            path_file = os.path.join(path, file_name)
            temp = pickle.load(open(path_file, "rb"))
            print(temp)
            df = df.append(temp)
            f += 1
    pickle.dump(df, open("generated/merged_result", "wb"))


def merge_analysis():
    df = pd.DataFrame()
    folder = "./generated/unmerged_analysis/"
    g = os.walk(folder)
    f = 0
    for path, dir_list, file_list in g:
        for file_name in tqdm(file_list, desc=str(f)):
            path_file = os.path.join(path, file_name)
            temp = pickle.load(open(path_file, "rb"))
            if f == 0:
                df = temp
            else:
                df["correct"] += temp["correct"]
                df["total"] += temp["total"]
            f += 1
    df["accuracy"] = df["correct"] / df["total"]
    pickle.dump(df, open("generated/merged_analysis", "wb"))


def merge_train_test():
    folders = ["generated/train_test/unmerged_x_test_vectorize", "generated/train_test/unmerged_x_train_vectorize",
               "generated/train_test/unmerged_y_test_vectorize", "generated/train_test/unmerged_y_train_vectorize"]
    out_names = ["generated/train_test/merged_x_test", "generated/train_test/merged_x_train",
                 "generated/train_test/merged_y_test", "generated/train_test/merged_y_train"]
    for folder, out_name in zip(folders, out_names):
        g = os.walk(folder)
        f = 0
        l = []
        for path, dir_list, file_list in g:
            for file_name in tqdm(file_list, desc=str(f)):
                path_file = os.path.join(path, file_name)
                temp = pickle.load(open(path_file, "rb"))
                l.extend(temp.tolist())
                f += 1
        out = np.array(l)
        pickle.dump(out, open(out_name, "wb"))


def merge():
    os.chdir("/scratch/scholar/lu677")
    # merge_result()
    # merge_analysis()
    merge_train_test()


merge()
