import requests
import os.path
import datetime
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np
import torch


class DatasetHandler:

    def __init__(self, filename: str = None, target_value="Close"):
        """
        init dataset Handler : Load data from url or file, and do the firsts transformations (to daily variance and round)
        :param filename: the filename that contains the data (if not, data are retrieved automatically online
        :param target_value: the value that we want to guess
        """
        # store features
        self.data = []

        self.filename = filename

        # parameters of the data
        self.round_floats = 5
        self.data_winsize = 0

        if self.filename:
            self.data = self.load_csv(self.filename)
        else:
            self.data = self.load_online()
        self.data = self.data.fillna(method='ffill')
        self.data = self.data.reset_index(drop=True)
        # self.data['daily_change'] = self.to_daily_var(self.data[target_value])
        # # increase the daily change to avoid it to be under 1, for MSE root square
        # self.data['daily_change'] = self.data['daily_change'] * 1000
        self.add_weekday()
        # round precision of daily change and eur_cad_rate
        self.data.round(self.round_floats)


        self.data_scaler = None

    def get_dataset(self):
        return self.data

    def get_features(self, features_list):
        print(f"Select features : {features_list}")
        return self.data[features_list]

    def normalize(self, data, margin=0.10):
        if self.data_scaler is None:
            self.data_scaler = MinMaxScaler()
            max_marg = data.max() * (1 + margin)
            min_marg = data.min() * (1 - margin)
            data_with_margin = data.append(min_marg, ignore_index=True)
            data_with_margin = data_with_margin.append(max_marg, ignore_index=True)
            self.data_scaler.fit(data_with_margin)
        return self.data_scaler.transform(data)


    def trunc_period(self, date_begin: str, date_end: str):
        """
        rescale dataset into a shorter period of time
        :param date_begin: format should be d-m-y
        :param date_end: format should be d-m-y
        """
        date_begin = pd.to_datetime(date_begin, format="%d-%m-%Y")
        date_end = pd.to_datetime(date_end, format="%d-%m-%Y")
        self.data = self.data[date_begin <= self.data.Date]
        self.data = self.data[self.data.Date <= date_end]

    def create_sequence(self, winsize: int, target_str: str):
        """
        add features for each element : the N previous values
        :param winsize: the size of the "memory" or sliding window. The number of precedent n features to guess the next
        :return:
        """
        self.data_winsize = winsize
        for win_i in range(1, winsize + 1):
            self.data['f' + str(win_i)] = list(np.zeros(win_i)) + [self.data[target_str].values[i - win_i]
                                                                   for i in
                                                                   range(win_i, len(self.data[target_str].values))]
        # remove the n first occurences because there is some zeros that are artificials
        self.data = self.data[winsize:]

    @staticmethod
    def train_test_split(train_size: float, data: list):
        """
        split the dataset to usable data to train the model.
        :param data: self.data if None, otherwise can contain any dataframe or
        :param train_size: the split of the dataset in percentage or int for train
        """
        if type(train_size) == int:
            split_point = train_size
        elif type(train_size) == float:
            split_point = int(len(data) * train_size)
        else:
            raise TypeError("train_size must be int or float")

        train = data[:split_point]
        test = data[split_point:]

        return train, test

    @staticmethod
    def df_augment(dataset, times: int, noise_lvl: float):
        """
        Augment dataset with little noise
        :param dataset: the list of features to augment
        :param times: how many times we augment
        :param noise_lvl: the level of the noise
        """

        aug_dataset = pd.DataFrame(dataset * times)
        aug_dataset = aug_dataset + np.random.normal(0, noise_lvl, aug_dataset.shape)
        aug_dataset = aug_dataset.values.tolist()
        aug_dataset.extend(dataset)
        return aug_dataset

    def add_weekday(self):
        self.data['weekday'] = self.data.Date.dt.dayofweek



    @staticmethod
    def load_csv(filename: str):
        df = pd.read_csv(filename)
        df.Date = pd.to_datetime(df.Date, format="%Y-%m-%d")
        # These 2 metrics are for stocks, not relevant for currencies
        df = df.drop(columns=["Volume", "Adj Close"])
        return df

    @staticmethod
    def load_online():
        # the yahoo url of the "download" data button, extending start/end period to be sure to capture everything
        url = "https://query1.finance.yahoo.com/v7/finance/download/EURCAD=X?period1=0&period2=9999999999999&interval=1d&events=history"

        today_date = datetime.datetime.today().strftime("%d-%m-%Y")
        filename = f"data/eur_cad_{today_date}.csv"
        if not os.path.isfile(filename):
            r = requests.get(url, allow_redirects=True)
            open(filename, 'wb').write(r.content)
        return DatasetHandler.load_csv(filename)

    @staticmethod
    def to_daily_var(data):
        """
        Difference between rate of yesterday and today. First value is 0 because we don't know
        :param data: the input pd serie
        :return: a new pd serie containing only the difference
        """
        array = pd.Series([0] + [data[i] - data[i - 1]  for i in range(1, len(data))])
        return array
    @staticmethod
    def to_tensor(features, target=None):
        """

        :param features:
        :param target:
        :return:
        """
        if target is not None:
            # Select the in and target features
            out_features = features[target].values.reshape(-1, 1)
            in_features = features.loc[:, features.columns != target].values

            # reshape to have N samples of an array of features
            in_features = in_features.reshape(len(in_features), 1, -1)

            in_features = torch.FloatTensor(in_features)
            out_features = torch.FloatTensor(out_features)
            # concatenate to have N samples of an array of features, an array of output
            tensor = [[in_features[i], out_features[i]] for i in range(0, len(in_features))]

            return tensor
        else:
            return torch.FloatTensor(features)

