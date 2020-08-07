import time
import numpy as np
import torch
import torch.nn as nn

import eurcad_model as models


def train(train_loader, learn_rate=0.00005, hidden_dim=256, hidden_nb=1, epochs=200, model_type="GRU"):
    # Setting hyperparameters
    input_dim = train_loader[0][0].shape[1]
    output_dim = 1
    batch_size = 128
    training_generator = torch.utils.data.DataLoader(train_loader, batch_size=batch_size, shuffle=True, drop_last=True)
    # Instantiating the models
    if model_type == "GRU":
        model = models.GruModel(input_size=input_dim, hidden_layer_nb=hidden_nb, hidden_layer_size=hidden_dim, output_size=output_dim)
    else:
        model = models.LstmModel(input_size=input_dim, hidden_layer_nb=hidden_nb, hidden_layer_size=hidden_dim, output_size=output_dim)
    model.to(model.device)

    # Choose loss, optimizer and metric
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    # set module state (for some objects it is needed)
    model.train()
    print(f"Training of {model_type}, batches of {batch_size} started...")
    epoch_times = []
    # Start training loop
    for epoch in range(1, epochs + 1):
        start_time = time.clock()
        # Reset the hidden states to zero
        hidden = model.init_hidden(batch_size)

        # metrics
        avg_loss = 0.
        counter = 0

        for x, label in training_generator:
            counter += 1
            if model_type == "GRU":
                hidden = hidden.data
            else:
                hidden = tuple([e.data for e in hidden])

            # set all gradients to zero because Pytorch accumulates gradients
            model.zero_grad()

            # forward pass
            out, hidden = model(x, hidden)

            # Compute the loss, gradients, and update the parameters by calling optimizer.step()
            loss = loss_function(out, label)
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            if counter % 200 == 0:
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter,
                                                                                           len(train_loader),
                                                                                           avg_loss / counter))
                # TODO evaluation each time to have the validation loss evolving
        current_time = time.clock()
        print("Epoch {}/{} Done, Total Loss: {}".format(epoch, epochs, avg_loss / len(train_loader)))
        epoch_times.append(current_time - start_time)
    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
    return model


def evaluate(model, test_x, test_y, label_scalers):
    model.eval()
    outputs = []
    targets = []
    start_time = time.clock()
    for i in test_x.keys():
        inp = torch.from_numpy(np.array(test_x[i]))
        labs = torch.from_numpy(np.array(test_y[i]))
        h = model.init_hidden(inp.shape[0])
        out, h = model(inp.to(model.device).float(), h)
        outputs.append(label_scalers[i].inverse_transform(out.cpu().detach().numpy()).reshape(-1))
        targets.append(label_scalers[i].inverse_transform(labs.numpy()).reshape(-1))
    print("Evaluation Time: {}".format(str(time.clock() - start_time)))
    sMAPE = 0
    for i in range(len(outputs)):
        sMAPE += np.mean(abs(outputs[i] - targets[i]) / (targets[i] + outputs[i]) / 2) / len(outputs)
    print("sMAPE: {}%".format(sMAPE * 100))
    return outputs, targets, sMAPE
