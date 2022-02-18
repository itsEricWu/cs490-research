import os
from tqdm import tqdm
import pandas as pd
import pickle


df = pd.DataFrame()
folder = "./generated/unmerged_analysis/"
g = os.walk(folder)
f = 0
df_out = pd.DataFrame(columns=["correct", "total", "accuracy", "label"])
for path, dir_list, file_list in g:
    for file_name in tqdm(file_list, desc=str(f)):
        path_file = os.path.join(path, file_name)
        temp = pickle.load(open(path_file, "rb"))
        for index, row in tqdm(temp.iterrows(), desc="record data"):
            if row["alpha"] == 0 and row["condition number"] == -1:
                df_out.loc[len(df_out.index)] = [row["correct"], row["total"],
                                                 row["accuracy"], path_file.split("_")[-1]]
pickle.dump(df_out, open("generated/original_image_analysis", "wb"))
