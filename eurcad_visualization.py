import matplotlib.pyplot as plt


def plot_raw(date, feature, title=""):
    """
    Just plot data
    :param date: the x axis, timeline
    :param feature: the y axis, price/variability
    :param title: title of the plot
    """
    plt.title(title)
    plt.plot(date, feature)
    plt.show()
