import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import eurcad_model as models


class Trainer:

    def __init__(self, model_type="GRU", model_params={}, logdir="rnn_eur_cad"):
        if len(model_params) == 0:
            model_params = {"learn_rate": 0.00005, "hidden_dim": 256, "hidden_nb": 1, "epochs": 200}

        self.learn_rate = model_params["learn_rate"]
        self.hidden_dim = model_params["hidden_dim"]
        self.hidden_nb = model_params["hidden_nb"]
        self.epochs = model_params["epochs"]
        self.model_type = model_type
        self.model = None
        self.input_dim = None
        self.output_dim = None
        self.batch_size = None
        self.optimizer = None
        self.loss_function = None

        is_cuda = torch.cuda.is_available()
        if is_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # create Tensorboard run and write the parameters of the run
        self.writer = SummaryWriter(f'runs/{logdir}')
        self.writer.add_text(self.model_type, f"Learning Rate : {self.learn_rate}")
        self.writer.add_text(self.model_type, f"Size of Hidden layer : {self.hidden_dim}")
        self.writer.add_text(self.model_type, f"Nb of Hidden layer : {self.hidden_nb}")
        self.writer.add_text(self.model_type, f"Nb of epochs : {self.epochs}")

    def train(self, train_set, test_set=None):
        # Setting hyperparameters
        self.input_dim = train_set[0][0].shape[1]
        self.output_dim = 1
        self.batch_size = 128

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True,
                                                   drop_last=True)
        train_labels = [x.detach().numpy()[0] for x in list(list(zip(*train_set))[1])]
        self.writer.add_text(self.model_type, f"Nb of inputs : {self.input_dim}")
        self.writer.add_text(self.model_type, f"Nb of Outputs : {self.output_dim}")
        self.writer.add_text(self.model_type, f"Batch Size : {self.batch_size}")

        if test_set is not None:
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set))
            test_labels = [x.detach().numpy()[0] for x in list(list(zip(*test_set))[1])]

        # Instantiating the models
        if self.model_type == "GRU":
            self.model = models.GruModel(self.device, input_size=self.input_dim, hidden_layer_nb=self.hidden_nb,
                                         hidden_layer_size=self.hidden_dim, output_size=self.output_dim)
        else:
            self.model = models.LstmModel(self.device, input_size=self.input_dim, hidden_layer_nb=self.hidden_nb,
                                          hidden_layer_size=self.hidden_dim, output_size=self.output_dim)
        self.model.to(self.device)

        # set up model graph to tensorboard
        self.writer.add_graph(self.model, (torch.rand(1, 1, self.input_dim), torch.rand(1, 1, self.hidden_dim)))

        # Choose loss, optimizer and metric
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learn_rate)
        self.loss_function = nn.MSELoss()


        # Notify network layers behaviour we are in test mode (batchnorm, dropout...)
        self.model.train()
        print(f"Training of {self.model_type}, batches of {self.batch_size} started...")
        epoch_times = []
        nb_loss = 0
        # Start training loop
        for epoch in range(1, self.epochs + 1):
            start_time = time.clock()
            # Reset the hidden states to zero
            hidden = self.model.init_hidden(self.batch_size)

            # metrics
            avg_loss = 0.
            avg_test_loss = 0.
            counter = 0

            for x, label in train_loader:
                counter += 1
                if self.model_type == "GRU":
                    hidden = hidden.data
                else:
                    hidden = tuple([e.data for e in hidden])

                # set all gradients to zero because Pytorch accumulates gradients
                self.model.zero_grad()

                # forward pass
                out, hidden = self.model(x, hidden)

                # Compute the loss, gradients, and update the parameters by calling optimizer.step()
                loss = self.loss_function(out, label)
                loss.backward()
                self.optimizer.step()

                avg_loss += loss.item()

                # Log the loss into tensorboard
                self.writer.add_scalar('Train loss', loss, nb_loss)
                if test_set is not None:
                    test_preds, test_loss = self.evaluate(test_loader)

                    self.writer.add_scalar('Test loss', test_loss, nb_loss)
                    avg_test_loss += test_loss
                nb_loss += 1
                if counter % 100 == 0:
                    print(
                        f"Epoch {epoch}......Step: {counter}/{len(train_loader)}....... Avg Loss for Epoch: {avg_loss / counter}...")
                    if test_set is not None:
                        print(f"Test : Avg Loss for Epoch: {avg_test_loss / counter}...")
            current_time = time.clock()

            print(f"Epoch {epoch}/{self.epochs} Done, Total Train Loss: {avg_loss / len(train_loader)}")
            if test_set is not None:
                print(f"Total Test Loss: {avg_test_loss / len(test_loader)}")
            epoch_times.append(current_time - start_time)
        print("Total Training Time: {} seconds".format(str(sum(epoch_times))))

        # Close tensorboard writer
        self.writer.close()

        return self.model

    def evaluate(self, test_loader):
        if type(test_loader) is list:
            test_loader = torch.utils.data.DataLoader(test_loader, batch_size=len(test_loader))

        # say to pytorch not to do the backpropagation
        with torch.no_grad():
            total_loss = 0
            self.model.eval()
            # calculate validation loss
            y_preds = []
            for i, data in enumerate(test_loader):
                X, y = data[0], data[1]
                y_pred = self.model(X, self.model.init_hidden(len(X)))[0]
                test_loss = self.loss_function(y_pred, y)
                total_loss += test_loss.item()
                y_preds.append(y_pred.detach().numpy().flatten())

            total_loss /= len(test_loader)
            # go back in train mode for the model
            self.model.train()
            return np.array(y_preds).flatten(), total_loss

    @staticmethod
    def get_smape(pred, truth):
        sMAPE = 0
        for i in range(len(pred)):
            sMAPE += np.mean(abs(pred[i] - truth[i]) / (truth[i] + pred[i]) / 2) / len(pred)

        return sMAPE