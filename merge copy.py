from copyreg import pickle
from tqdm import tqdm
import os
import pickle
import pandas as pd


def merge_result():
    df = pd.DataFrame()
    folder = "./asd/unmerged_result/"
    g = os.walk(folder)
    f = 0
    for path, dir_list, file_list in g:
        for file_name in tqdm(file_list, desc=str(f)):
            path_file = os.path.join(path, file_name)
            temp = pickle.load(open(path_file, "rb"))
            print(temp)
            df = df.append(temp)
            f += 1
    pickle.dump(df, open("asd/merged_result", "wb"))


def merge_analysis():
    df = pd.DataFrame()
    folder = "./asd/unmerged_analysis/"
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
    pickle.dump(df, open("asd/merged_analysis", "wb"))


merge_analysis()
