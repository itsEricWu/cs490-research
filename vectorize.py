import sys
import pickle
import numpy as np


def vectorize_x(file_name, out_name):
    data = pickle.load(open(file_name, "rb"))
    scaler = pickle.load(open("generated/scaler", "rb"))
    data.drop("original image name", axis=1)
    data.drop("actual label for processed", axis=1)
    data.drop("predicted label for processed", axis=1)
    out_list = []
    for index, row in data.iterrows():
        temp = []
        r = []
        r.extend(row["image"])
        temp.extend(row["v"].flatten())
        temp.append(row["alpha"])
        temp.append(row["condition number"])
        temp = scaler.transform([temp])
        r.extend(temp.flatten())
        out_list.append(r)
    arr = np.array(out_list)
    pickle.dump(arr, open(out_name, "wb"))


def vectorize_y(file_name, out_name):
    data = pickle.load(open(file_name, "rb"))
    out_list = data
    arr = np.array(out_list)
    pickle.dump(arr, open(out_name, "wb"))


SD = "/scratch/scholar/lu677/"
x_test = SD + "generated/train_test/unmerged_x_test/" + sys.argv[1]
x_train = SD + "generated/train_test/unmerged_x_train/" + sys.argv[1]
y_test = SD + "generated/train_test/unmerged_y_test/" + sys.argv[1]
y_train = SD + "generated/train_test/unmerged_y_train/" + sys.argv[1]
x_test_out = SD + "generated/train_test/unmerged_x_test_normalize/" + sys.argv[1]
x_train_out = SD + "generated/train_test/unmerged_x_train_normalize/" + sys.argv[1]
y_test_out = SD + "generated/train_test/unmerged_y_test_normalize/" + sys.argv[1]
y_train_out = SD + "generated/train_test/unmerged_y_train_normalize/" + sys.argv[1]
vectorize_x(x_test, x_test_out)
vectorize_x(x_train, x_train_out)
vectorize_y(y_test, y_test_out)
vectorize_y(y_train, y_train_out)
