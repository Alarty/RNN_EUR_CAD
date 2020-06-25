import preprocessor
import viz


data_loader = preprocessor.DataLoader(filename="data/eur_cad_1999-2020-daily-closing.csv")
data = data_loader.get_data()
viz.plot_raw(data.date, data.eur_cad_rate)
viz.plot_raw(data.date[-800:], data.eur_cad_rate[-800:])
