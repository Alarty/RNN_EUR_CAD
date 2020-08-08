import time
import numpy as np
import torch
import torch.nn as nn

import eurcad_model as models


class Trainer:

    def __init__(self, learn_rate=0.00005, hidden_dim=256, hidden_nb=1, epochs=200, model_type="GRU"):

        self.learn_rate = learn_rate
        self.hidden_dim = hidden_dim
        self.hidden_nb = hidden_nb
        self.epochs = epochs
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

    def train(self, train_set, test_set=None):
        # Setting hyperparameters
        self.input_dim = train_set[0][0].shape[1]
        self.output_dim = 1
        self.batch_size = 128
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True,
                                                   drop_last=True)
        if test_set is not None:
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=1)
        # Instantiating the models
        if self.model_type == "GRU":
            self.model = models.GruModel(self.device, input_size=self.input_dim, hidden_layer_nb=self.hidden_nb,
                                         hidden_layer_size=self.hidden_dim, output_size=self.output_dim)
        else:
            self.model = models.LstmModel(self.device, input_size=self.input_dim, hidden_layer_nb=self.hidden_nb,
                                          hidden_layer_size=self.hidden_dim, output_size=self.output_dim)
        self.model.to(self.device)

        # Choose loss, optimizer and metric
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learn_rate)
        self.loss_function = nn.MSELoss()

        # Notify network layers behaviour we are in test mode (batchnorm, dropout...)
        self.model.train()
        print(f"Training of {self.model_type}, batches of {self.batch_size} started...")
        epoch_times = []
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
                if test_set is not None:
                    test_loss = self.validate(test_loader)
                    avg_test_loss += test_loss
                if counter % 100 == 0:
                    if test_set is not None:
                        print(f"Epoch {epoch}......Step: {counter}/{len(train_loader)}....... Avg Loss for Epoch: {avg_loss / counter}... Avg Test Loss: {avg_test_loss / counter}")
                    else:
                        print(f"Epoch {epoch}......Step: {counter}/{len(train_loader)}....... Avg Loss for Epoch: {avg_loss / counter}")

                    # TODO evaluation each time to have the validation loss evolving
            current_time = time.clock()
            if test_set is not None:
                print(f"Epoch {epoch}/{self.epochs} Done, Total Train Loss: {avg_loss / len(train_loader)}, Total Test Loss: {avg_test_loss / len(train_loader)}")
            else:
                print(f"Epoch {epoch}/{self.epochs} Done, Total Train Loss: {avg_loss / len(train_loader)}")
            epoch_times.append(current_time - start_time)
        print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
        return self.model

    def validate(self, test_loader):
        # say to pytorch not to do the backpropagation
        with torch.no_grad():
            total_loss = 0
            self.model.eval()
            # calculate validation loss
            for i, data in enumerate(test_loader):
                X, y = data[0], data[1]
                y_pred = self.model(X, self.model.init_hidden(1))[0]
                test_loss = self.loss_function(y_pred, y)
                total_loss += test_loss.item()

            total_loss /= len(test_loader)

            # go back in train mode for the model
            self.model.train()
            return total_loss


    def evaluate(self, model, test_x, test_y, label_scalers):
        model.eval()
        outputs = []
        targets = []
        start_time = time.clock()
        for i in test_x.keys():
            inp = torch.from_numpy(np.array(test_x[i]))
            labs = torch.from_numpy(np.array(test_y[i]))
            h = model.init_hidden(inp.shape[0])
            out, h = model(inp.to(self.device).float(), h)
            outputs.append(label_scalers[i].inverse_transform(out.cpu().detach().numpy()).reshape(-1))
            targets.append(label_scalers[i].inverse_transform(labs.numpy()).reshape(-1))
        print("Evaluation Time: {}".format(str(time.clock() - start_time)))
        sMAPE = 0
        for i in range(len(outputs)):
            sMAPE += np.mean(abs(outputs[i] - targets[i]) / (targets[i] + outputs[i]) / 2) / len(outputs)
        print("sMAPE: {}%".format(sMAPE * 100))
        return outputs, targets, sMAPE
