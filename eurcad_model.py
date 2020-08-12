import torch
import torch.nn as nn


class RnnModel(nn.Module):
    def __init__(self, device, input_size=1, hidden_layer_nb=1, hidden_layer_size=100, output_size=1, drop_prob=0.2, load=None):
        """
        Init of the Model instance
        :param input_size: Number of features for the input for 1 time moment
        :param hidden_layer_nb: Number of hidden layer
        :param hidden_layer_size: Number of neuron in the hidden layer
        :param output_size: the number of features we want to predict
        :param load: If this is a new model to train : leave empty. If this is a old model to load, path of the pkl file
        """
        super(RnnModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.hidden_layer_nb = hidden_layer_nb
        self.input_size = input_size
        self.output_size = output_size
        self.drop_prob = drop_prob
        self.device = device

class GruModel(RnnModel):
    def __init__(self, device, input_size=1, hidden_layer_nb=1, hidden_layer_size=100, output_size=1, drop_prob=0.2):
        super(GruModel, self).__init__(device, input_size, hidden_layer_nb, hidden_layer_size, output_size, drop_prob)
        self.gru = nn.GRU(input_size, hidden_layer_size, hidden_layer_nb, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_layer_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:, -1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.hidden_layer_nb, batch_size, self.hidden_layer_size).zero_().to(self.device)
        return hidden


class LstmModel(RnnModel):
    def __init__(self, device, input_size=1, hidden_layer_nb=1, hidden_layer_size=100, output_size=1, drop_prob=0.2):
        super(LstmModel, self).__init__(device, input_size, hidden_layer_nb, hidden_layer_size, output_size, drop_prob)
        self.hidden_dim = hidden_layer_size
        self.n_layers = hidden_layer_nb

        self.lstm = nn.LSTM(input_size, hidden_layer_size, hidden_layer_nb, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_layer_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.lstm(x, h)
        out = self.fc(self.relu(out[:, -1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.hidden_layer_nb, batch_size, self.hidden_layer_size).zero_().to(self.device),
                  weight.new(self.hidden_layer_nb, batch_size, self.hidden_layer_size).zero_().to(self.device))
        return hidden


class Arima:
    def __init__(self):
        pass


