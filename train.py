import datetime

import eurcad_dataset

dataset = eurcad_dataset.DatasetHandler(filename="data/eur_cad_1999-2020-daily-closing.csv")
dataset.trunc_period("01-01-2018", datetime.datetime.today().strftime("%d-%m-%Y"))
dataset.creat_window_features(4)
dataset.train_test_split(0.8)
dataset.data_augment(3, 0.0001)

print("End")
