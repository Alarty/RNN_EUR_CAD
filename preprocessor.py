import pandas as pd


class DataLoader:

    def __init__(self, origin_url=None, filename=None):
        self.data = []
        self.origin_url = origin_url
        self.filename = filename
        if self.filename:
            self.data = self.load_csv(self.filename)
        elif self.origin_url:
            self.data = self.load_webscrap(self.origin_url)
        else:
            raise AttributeError("A filename or a url must be provided to retrieve data")

    def get_data(self):
        return self.data

    @staticmethod
    def load_csv(filename):
        df = pd.read_csv(filename)
        df.date = pd.to_datetime(df.date, format="%d-%m-%Y")
        return df

    @staticmethod
    def load_webscrap(url):
        # https://www.ofx.com/en-au/forex-news/historical-exchange-rates/
        raise NotImplementedError
