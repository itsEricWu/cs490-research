import pickle
import pandas as pd
import numpy as np
import tqdm as tqdm


def old_result():
    df = pickle.load(open("result", "rb"))
    out_df = pd.DataFrame(columns=["type", "correct", "total", "accuracy", "classfication", "alpha"])
    original_count = 0
    correct_count = 0
    epsilons = [0, .05, .1, .2, .4, .8]
    classfication = list(range(0, 19))
    type = ["original", "attacked", "v matrix", "attacked v matrix"]

    # creating rows
    for c in classfication:
        for a in epsilons:
            if a == 0:
                out_df.loc[len(out_df.index)] = [type[0], 0, 0, 0, c, a]
                out_df.loc[len(out_df.index)] = [type[2], 0, 0, 0, c, a]
            else:
                out_df.loc[len(out_df.index)] = [type[1], 0, 0, 0, c, a]
                out_df.loc[len(out_df.index)] = [type[3], 0, 0, 0, c, a]
    print("row created")
    # record data for each row
    for index, row in df.iterrows():
        a = row["alpha"]
        c = row["actual label for processed"]
        t = ""
        if a == 0 and np.array_equal(np.array(row["v"]), np.identity(3)):
            t = type[0]
        if a != 0 and np.array_equal(np.array(row["v"]), np.identity(3)):
            t = type[1]
        if a == 0 and not np.array_equal(np.array(row["v"]), np.identity(3)):
            t = type[2]
        if a != 0 and not np.array_equal(np.array(row["v"]), np.identity(3)):
            t = type[3]
        out_df.loc[((out_df.type == t) & (out_df.classfication == c) & (out_df.alpha == a)), "total"] += 1
        if row["actual label for processed"] == row["predicted label for processed"]:
            out_df.loc[((out_df.type == t) & (out_df.classfication == c) & (out_df.alpha == a)), "correct"] += 1
    print("row inserted")

    # calculate accuracy for each row
    for index, row in out_df.iterrows():
        print(row)
        row["accuracy"] = row["correct"] / row["total"]
    print(out_df)
    pickle.dump(out_df, open("analysis", "wb"))

def new_result():
    df = pickle.load(open("generated/changed_v_result", "rb"))
    out_df = pd.DataFrame(columns=["correct", "total", "accuracy", "alpha", "condition number"])
    epsilons = pickle.load(open("generated/epsilons", "rb"))
    condition_list = pickle.load(open("generated/condition_list", "rb"))

    # creating rows
    for c in condition_list:
        for a in epsilons:
            out_df.loc[len(out_df.index)] = [0, 0, 0, a, c]

    # record data for each row
    for index, row in df.iterrows():
        a = row["alpha"]
        c = row["condition number"]
        out_df.loc[((out_df["condition number"] == c) & (out_df["alpha"] == a)), "total"] += 1
        if row["actual label for processed"] == row["predicted label for processed"]:
            out_df.loc[((out_df["condition number"] == c) & (out_df["alpha"] == a)), "correct"] += 1

    # calculate accuracy for each row
    for index, row in out_df.iterrows():
        print(row)
        row["accuracy"] = row["correct"] / row["total"]
    pickle.dump(out_df, open("generated/changed_v_analysis", "wb"))

new_result()
