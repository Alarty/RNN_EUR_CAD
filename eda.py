import dataset
import viz
import datetime

data_loader = dataset.DataHandler(filename="data/eur_cad_1999-2020-daily-closing.csv")
data_loader.filter_data_by_date("01-01-2018", datetime.datetime.today().strftime("%d-%m-%Y"))
data_loader.creat_window_features(4)
# viz.plot_raw(data_loader.data.date, data_loader.data.eur_cad_rate)
# viz.plot_raw(data_loader.data.date, data_loader.data.daily_change)
data_loader.train_test_split(0.8)
data_loader.data_augment_train_test(3, 0.0001)
print("End")
