import matplotlib.pyplot as plt

def plot_raw(date, price):
    plt.title("Raw plot, day vs exchange rate EUR-CAD")
    plt.plot(date, price)
    plt.show()
