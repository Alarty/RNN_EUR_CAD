import dataset
import viz
import datetime

data_loader = dataset.DataHandler(filename="data/eur_cad_1999-2020-daily-closing.csv")
data_loader.filter_data_by_date("01-01-2018", datetime.datetime.today().strftime("%d-%m-%Y"))

data_loader.train_test_split(4, 0.8)

data = data_loader.get_dataset()
viz.plot_raw(data.date, data.eur_cad_rate)
