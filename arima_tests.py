import copy
import datetime

from tqdm import tqdm
from pandas import Index

import eurcad_dataset as ds
from matplotlib import pyplot as plt
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# PARAMETERS
# dataset parameters
data_start_date = "01-01-2020"
data_end_date = datetime.datetime.today().strftime("%d-%m-%Y")
logdir_timestamp = datetime.datetime.today().strftime("%y-%m-%d_%H-%M-%S")
# Number of features to prediction the nth one
sequence_size = 7

# Model parameters
model_params = {"learn_rate": 0.0001, "hidden_dim": 128, "hidden_nb": 1, "epochs": 400, "data_augment": False}
model_type = "ARIMA"

model_is_nn = model_type in ["GRU", "LSTM"]

print(f"Run : {model_type}_data_{logdir_timestamp}")

# data gathering
ds_target_var = "Close"
dataset = ds.DatasetHandler(target_value=ds_target_var)
dataset.trunc_period(data_start_date, data_end_date)
features_str = ["Close", "Date"]
df = dataset.get_features(features_str)

df.set_index('Date', inplace=True)

train_data, test_data = dataset.train_test_split(0.90, df)
train_data = train_data["Close"]
test_data = test_data["Close"]

train_data_mod_1_lag = copy.deepcopy(train_data)
train_data_mod_no_lag = copy.deepcopy(train_data)
pred_no_lag = list()
pred_1_lag = list()

for t in tqdm(range(len(test_data))):
    model_1_lag = ARIMA(train_data_mod_1_lag, order=(5, 1, 0))
    model_no_lag = ARIMA(train_data_mod_no_lag, order=(5, 1, 0))
    model_1_lag = model_1_lag.fit()
    model_no_lag = model_no_lag.fit()
    output_1_lag = model_1_lag.forecast()
    output_no_lag = model_no_lag.forecast()
    yhat_1_lag = output_1_lag[0]
    yhat_no_lag = output_no_lag[0]
    pred_1_lag.append(yhat_1_lag)
    pred_no_lag.append(yhat_no_lag)
    train_data_mod_1_lag[test_data.index[t]] = test_data[t]
    train_data_mod_no_lag[test_data.index[t]] = yhat_no_lag

print("End of Loop")
print(pred_1_lag)
print(pred_no_lag)
error_1_lag = mean_squared_error(test_data, pred_1_lag)
error_no_lag = mean_squared_error(test_data, pred_no_lag)
print('Test MSE 1 lag: %.3f' % error_1_lag)
print('Test MSE no lag: %.3f' % error_no_lag)
full_data = train_data_mod_1_lag
pred_1_lag = pd.Series(pred_1_lag, index=test_data.index)
pred_no_lag = pd.Series(pred_no_lag, index=test_data.index)

plt.plot(full_data, label="Truth")
plt.plot(pred_1_lag, label="Pred with 1d lag")
plt.plot(pred_no_lag, label="Pred")
plt.legend(loc="upper left")
plt.show()
