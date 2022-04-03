import splitfolders

input_folder = "Train/"

splitfolders.ratio(input_folder, output="TrainVal", seed=42, ratio=(.8, .2), group_prefix=None)
