from sklearn.model_selection import train_test_split
import pickle
from preprocess import Preprocess
import sys

file_name = "./generated/unmerged_result/changed_v_result_" + sys.argv[1]
data = pickle.load(open(file_name, "rb"))
image_matrix = []
for index, row in data.iterrows():
    image_matrix.append(Preprocess.linearize_image(
        row["original image name"], row["v"], row["alpha"]))
data["image"] = image_matrix
y = data["probability"]
x = data.drop("probability", axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
xtest_path = "generated/train_test/unmerged_x_test/" + sys.argv[1]
xtrain_path = "generated/train_test/unmerged_x_train/" + sys.argv[1]
ytest_path = "generated/train_test/unmerged_y_test/" + sys.argv[1]
ytrain_path = "generated/train_test/unmerged_y_train/" + sys.argv[1]
pickle.dump(x_test, open(xtest_path, "wb"))
pickle.dump(x_train, open(xtrain_path, "wb"))
pickle.dump(y_test, open(ytest_path, "wb"))
pickle.dump(y_train, open(ytrain_path, "wb"))
