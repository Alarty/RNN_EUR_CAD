import datetime

import eurcad_dataset
import eurcad_visualization

dataset = eurcad_dataset.DatasetHandler()

dataset.trunc_period("01-01-2000", datetime.datetime.today().strftime("%d-%m-%Y"))
eurcad_visualization.plot_raw(dataset.data.Date, dataset.data.Close, "Raw exchange rate across time")
eurcad_visualization.plot_raw(dataset.data.Date, dataset.data.daily_change, "Closing daily variation across time")
