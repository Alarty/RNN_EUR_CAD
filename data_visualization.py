import datetime

import eurcad_dataset
import eurcad_visualization

dataset = eurcad_dataset.DatasetHandler(filename="data/eur_cad_1999-2020-daily-closing.csv")

dataset.trunc_period("01-01-2018", datetime.datetime.today().strftime("%d-%m-%Y"))
eurcad_visualization.plot_raw(dataset.data.date, dataset.data.eur_cad_rate, "Raw exchange rate across time")
eurcad_visualization.plot_raw(dataset.data.date, dataset.data.daily_change, "Closing daily variation across time")
