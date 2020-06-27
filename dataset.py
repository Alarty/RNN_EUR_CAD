import pandas as pd
import numpy as np


class DataHandler:

    def __init__(self, origin_url: str = None, filename: str = None):
        self.data = []
        self.train = None
        self.test = None
        self.origin_url = origin_url
        self.filename = filename
        self.round_floats = 5
        self.data_winsize = 0
        if self.filename:
            self.data = self.load_csv(self.filename)
        elif self.origin_url:
            self.data = self.load_webscrap(self.origin_url)
        else:
            raise AttributeError("A filename or a url must be provided to retrieve data")
        self.data['daily_change'] = self.exchange_rate_to_daily_change(self.data.eur_cad_rate)
        self.add_date_info()
        # round precision of daily change and eur_cad_rate
        self.data.round(self.round_floats)

    def get_dataset(self):
        return self.data

    def get_train_test(self):
        return self.train, self.test

    def filter_data_by_date(self, date_begin: str, date_end: str):
        """
        rescale dataset into a shorter period of time
        :param date_begin: format should be d-m-y
        :param date_end: format should be d-m-y
        """
        date_begin = pd.to_datetime(date_begin, format="%d-%m-%Y")
        date_end = pd.to_datetime(date_end, format="%d-%m-%Y")
        self.data = self.data[date_begin <= self.data.date]
        self.data = self.data[self.data.date <= date_end]

    def creat_window_features(self, winsize: int):
        """

        :param winsize: the size of the "memory" or sliding window. The number of precedent n features to guess the next
        :return:
        """
        self.data_winsize = winsize
        for win_i in range(1, winsize + 1):
            self.data['f' + str(win_i)] = list(np.zeros(win_i)) + [self.data["daily_change"].values[i - win_i]
                                                                   for i in range(win_i, len(self.data["daily_change"].values))]
        # remove the n first occurences because there is some zeros that are artificials
        self.data = self.data[winsize:]

    def train_test_split(self, train_size: float):
        """
        split the dataset to usable data to train the model.
        :param train_size: the split of the dataset in percentage or int for train
        """
        if type(train_size) == int:
            split_point = train_size
        elif type(train_size) == float:
            split_point = int(len(self.data) * train_size)
        else:
            raise TypeError("train_size must be int or float")

        self.train = self.data[:split_point]
        self.test = self.data[split_point:]

        self.train = self.train.drop(['date', 'eur_cad_rate'], axis=1)
        self.test = self.test.drop(['date', 'eur_cad_rate'], axis=1)

    def data_augment_train_test(self, replicate: int, noise_lvl: float):
        for dataset in [self.train, self.test]:

            dataset = pd.concat([dataset] * replicate, ignore_index=True)
            random_seq = np.random.normal(0, noise_lvl, len(dataset))
            dataset.daily_change += random_seq
            for feat_num in range(0, self.data_winsize):
                random_seq = np.random.normal(0, noise_lvl, len(dataset))
                dataset["f"+str(feat_num+1)] += random_seq
        self.data.round(self.round_floats)

    def add_date_info(self):
        self.data['weekday'] = self.data.date.dt.dayofweek

    @staticmethod
    def load_csv(filename: str):
        df = pd.read_csv(filename)
        df.date = pd.to_datetime(df.date, format="%d-%m-%Y")
        return df

    @staticmethod
    def load_webscrap(url: str):
        # https://www.ofx.com/en-au/forex-news/historical-exchange-rates/
        raise NotImplementedError

    @staticmethod
    def exchange_rate_to_daily_change(rate: object):
        """
        Difference between rate of yesterday and today. First value is 0 because we don't know
        """
        return pd.Series([0] + [rate[i] - rate[i - 1] for i in range(1, len(rate))])
