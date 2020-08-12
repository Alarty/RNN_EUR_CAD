import datetime

import eurcad_dataset as ds
import eurcad_trainer as trainer
import eurcad_visualization as viz

# PARAMETERS
# dataset parameters
data_start_date = "01-01-2018"
data_end_date = datetime.datetime.today().strftime("%d-%m-%Y")
# Number of features to prediction the nth one
sequence_size = 4

# Model parameters
model_params = {"learn_rate": 0.00005, "hidden_dim": 256, "hidden_nb": 1, "epochs": 200}
model_type = "GRU"


# data gathering
dataset = ds.DatasetHandler(filename="data/eur_cad_1999-2020-daily-closing.csv")
dataset.trunc_period(data_start_date, data_end_date)
dataset.create_sequence(sequence_size)

# feature gathering
features_str = [f"f{i}" for i in range(1, sequence_size + 1)] + ['daily_change']
features = dataset.get_features(features_str)
features = dataset.to_tensor(features, target="daily_change")
train_data, test_data = dataset.train_test_split(0.8, features)

# training of the model

trainer = trainer.Trainer(model_type=model_type, model_params=model_params, logdir=f"{model_type}_data_end_date_{data_end_date}")
trainer.train(train_data)

# evaluation of the model
test_labels = [x.detach().numpy()[0] for x in list(list(zip(*test_data))[1])]
test_preds, test_loss = trainer.evaluate(test_data)
viz.plot_difference(test_preds, test_labels, "Difference between test prediction and truth")

print("End")
