import datetime
from torch.utils.tensorboard import SummaryWriter

import eurcad_dataset as ds
import eurcad_trainer as trainer
import eurcad_visualization as viz

# PARAMETERS
# dataset parameters
data_start_date = "01-01-2000"
data_end_date = datetime.datetime.today().strftime("%d-%m-%Y")
logdir_timestamp = datetime.datetime.today().strftime("%y-%m-%d_%H-%M-%S")
# Number of features to prediction the nth one
sequence_size = 7


# Model parameters
model_params = {"learn_rate": 0.0001, "hidden_dim": 128, "hidden_nb": 1, "epochs": 400, "data_augment":False}
model_type = "ARIMA"

model_is_nn = model_type in ["GRU", "LSTM"]

print(f"Run : {model_type}_data_{logdir_timestamp}")

# data gathering
ds_target_var = "Close"
dataset = ds.DatasetHandler(target_value=ds_target_var)
dataset.trunc_period(data_start_date, data_end_date)
dataset.create_sequence(sequence_size, ds_target_var)

# feature gathering
features_str = [f"f{i}" for i in range(1, sequence_size + 1)] + [ds_target_var]
features = dataset.get_features(features_str)
features[features_str] = dataset.normalize(features)

if model_is_nn:
    features = dataset.to_tensor(features, target=ds_target_var)

train_data, test_data = dataset.train_test_split(0.8, features)
if model_params["data_augment"] is True:
    train_data = dataset.df_augment(train_data, 2, 0.1)

# create Tensorboard run and write the parameters of the run
writer = SummaryWriter(f'runs/{model_type}_data_{logdir_timestamp}')
writer.add_text(model_type, f"Database start : {data_start_date}")
writer.add_text(model_type, f"Database end : {data_end_date}")

# training of the model
trainer = trainer.Trainer(is_nn=model_is_nn, model_type=model_type, model_params=model_params, writer=writer)
trainer.train(train_data, test_data)

# evaluation of the model
test_labels = [x.detach().numpy()[0] for x in list(list(zip(*test_data))[1])]
test_preds, test_loss = trainer.evaluate(test_data)
for i in range(len(test_preds)):
    writer.add_scalars('Difference between test prediction and truth',
                       {"Predictions": test_preds[i], "Truth": test_labels[i]}, i)

print("End")
