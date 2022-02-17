from copyreg import pickle
from tqdm import tqdm
import os
import pickle
import pandas as pd


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


def merge():
    merge_result()
    merge_analysis()


merge()
