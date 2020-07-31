import pandas as pd
import numpy as np


class DatasetHandler:

    def __init__(self, origin_url: str = None, filename: str = None):
        """
        init dataset Handler : Load data from url or file, and do the firsts transformations (to daily variance and round)
        :param origin_url: the web URL that contains the data
        :param filename: the filename that contains the data
        """
        # store features
        self.data = []
        self.train = None
        self.test = None

        # data input
        self.origin_url = origin_url
        self.filename = filename

        # parameters of the data
        self.round_floats = 5
        self.data_winsize = 0

        if self.filename:
            self.data = self.load_csv(self.filename)
        elif self.origin_url:
            self.data = self.load_webscrap(self.origin_url)
        else:
            raise AttributeError("A filename or a url must be provided to retrieve data")

        self.data['daily_change'] = self.to_daily_var(self.data.eur_cad_rate)
        self.add_weekday()
        # round precision of daily change and eur_cad_rate
        self.data.round(self.round_floats)

    def get_dataset(self):
        return self.data

    def get_train_test(self):
        return self.train, self.test

    def trunc_period(self, date_begin: str, date_end: str):
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
        add features for each element : the N previous values
        :param winsize: the size of the "memory" or sliding window. The number of precedent n features to guess the next
        :return:
        """
        self.data_winsize = winsize
        for win_i in range(1, winsize + 1):
            self.data['f' + str(win_i)] = list(np.zeros(win_i)) + [self.data["daily_change"].values[i - win_i]
                                                                   for i in
                                                                   range(win_i, len(self.data["daily_change"].values))]
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

    def data_augment(self, times: int, noise_lvl: float):
        """
        Augment dataset with little noise
        :param times: how many times we augment
        :param noise_lvl: the level of the noise
        """
        for dataset in [self.train, self.test]:

            dataset = pd.concat([dataset] * times, ignore_index=True)
            random_seq = np.random.normal(0, noise_lvl, len(dataset))
            dataset.daily_change += random_seq
            for feat_num in range(0, self.data_winsize):
                random_seq = np.random.normal(0, noise_lvl, len(dataset))
                dataset["f" + str(feat_num + 1)] += random_seq
        self.data.round(self.round_floats)

    def add_weekday(self):
        self.data['weekday'] = self.data.date.dt.dayofweek



    @staticmethod
    def load_csv(filename: str):
        df = pd.read_csv(filename)
        df.date = pd.to_datetime(df.date, format="%d-%m-%Y")
        return df

    @staticmethod
    def load_webscrap(url: str):
        # https://www.ofx.com/en-au/forex-news/historical-exchange-rates/
        # https://finance.yahoo.com/quote/EURCAD%3DX/history?p=EURCAD%3DX
        raise NotImplementedError

    @staticmethod
    def to_daily_var(data: object):
        """
        Difference between rate of yesterday and today. First value is 0 because we don't know
        :param data: the input pd serie
        :return: a new pd serie containing only the difference
        """
        return pd.Series([0] + [data[i] - data[i - 1] for i in range(1, len(data))])
