import datetime

import eurcad_dataset
import eurcad_trainer as trainer

sequence_size = 4

dataset = eurcad_dataset.DatasetHandler(filename="data/eur_cad_1999-2020-daily-closing.csv")
dataset.trunc_period("01-01-2018", datetime.datetime.today().strftime("%d-%m-%Y"))
dataset.create_sequence(sequence_size)

features_str = [f"f{i}" for i in range(1, sequence_size + 1)] + ['daily_change']
features = dataset.get_features(features_str)
features = dataset.to_tensor(features, target="daily_change")
train_data, test_data = dataset.train_test_split(0.8, features)

trainer = trainer.Trainer()
trainer.train(train_data, test_data)
print("End")
