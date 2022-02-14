import pickle
import pandas as pd
import numpy as np

df = pickle.load(open("result", "rb"))

out_df = pd.DataFrame(columns=["type", "correct", "total", "accuracy", "classfication", "alpha"])
original_count = 0
correct_count = 0
epsilons = [0, .05, .1, .2, .4, .8]
classfication = list(range(0, 18))
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

#calculate accuracy for each row
for index, row in out_df.iterrows():
    print(row)
    row["accuracy"] = row["correct"] / row["total"]
print(out_df)
pickle.dump(df, open("analysis", "wb"))
