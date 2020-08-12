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


def plot_difference(test_preds, test_labels, title=""):
    assert len(test_preds) == len(test_labels)
    plt.title(title)
    plt.plot(range(0, len(test_preds)), test_preds)
    plt.plot(range(0, len(test_labels)), test_labels)
    plt.show()