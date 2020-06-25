import copy
import pandas as pd
import numpy as np

class DataHandler:

    def __init__(self, origin_url=None, filename=None):
        self.data = []
        self.train = None
        self.test = None
        self.origin_url = origin_url
        self.filename = filename
        if self.filename:
            self.data = self.load_csv(self.filename)
        elif self.origin_url:
            self.data = self.load_webscrap(self.origin_url)
        else:
            raise AttributeError("A filename or a url must be provided to retrieve data")
        self.data['daily_change'] = self.exchange_rate_to_daily_change(self.data.eur_cad_rate)
        # round precision of daily change and eur_cad_rate
        self.data.round(5)

    def get_dataset(self):
        return self.data

    def get_train_test(self):
        return self.train, self.test

    def filter_data_by_date(self, date_begin, date_end):
        """
        rescale dataset into a shorter period of time
        :param date_begin: format should be d-m-y
        :param date_end: format should be d-m-y
        """
        date_begin = pd.to_datetime(date_begin, format="%d-%m-%Y")
        date_end = pd.to_datetime(date_end, format="%d-%m-%Y")
        self.data = self.data[date_begin <= self.data.date]
        self.data = self.data[self.data.date <= date_end]

    @staticmethod
    def load_csv(filename):
        df = pd.read_csv(filename)
        df.date = pd.to_datetime(df.date, format="%d-%m-%Y")
        return df

    @staticmethod
    def load_webscrap(url):
        # https://www.ofx.com/en-au/forex-news/historical-exchange-rates/
        raise NotImplementedError

    @staticmethod
    def exchange_rate_to_daily_change(rate):
        """
        Difference between rate of yesterday and today. First value is 0 because we don't know
        """
        return pd.Series([0] + [rate[i] - rate[i-1] for i in range(1, len(rate))])

    def train_test_split(self, winsize: int, train_size):
        """
        split the dataset to usable data to train the model.
        :param winsize: the size of the "memory" or sliding window. The number of precedent n features to guess the next
        :param train_size: the split of the dataset in percentage or int for train
        """
        if type(train_size) == int:
            split_point = train_size
        elif type(train_size) == float:
            split_point = int(len(self.data) * train_size)
        else:
            raise TypeError("train_size must be int or float")
        data_copied = copy.deepcopy(self.data)
        for win_i in range(1, winsize+1):
            data_copied['f'+str(win_i)] = list(np.zeros(win_i)) + [data_copied["daily_change"].values[i-win_i]
                                                                   for i in range(win_i, len(data_copied["daily_change"].values))]
        # remove the n first occurences because there is some zeros that are artificials
        data_copied = data_copied[winsize:]
        self.train = data_copied[:split_point]
        self.test = data_copied[split_point:]

        print(len(self.train))
        print(len(self.test))

        return self.train, self.test
