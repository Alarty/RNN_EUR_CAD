# Bibliography

## GRU vs LSTM
- GRU pros :
    - Train faster so less training data required
    - Simpler architecture, easier to tune model architecture
    - More "efficient"
    - Best results with few training
- LSTM pros : 
    - More sophisticate
    - Longer/Harder to train
    - Better handle of long term relation
    - More controllable
    - Best results if fine tune  

## LSTM best explaination : https://colah.github.io/posts/2015-08-Understanding-LSTMs/

##https://lilianweng.github.io/lil-log/2017/07/08/predict-stock-prices-using-RNN-part-1.html#define-graph

LSTM Definitions : 

    lstm_size: number of units in one LSTM layer.
    num_layers: number of stacked LSTM layers.
    keep_prob: percentage of cell units to keep in the dropout operation.
    init_learning_rate: the learning rate to start with.
    learning_rate_decay: decay ratio in later training epochs.
    init_epoch: number of epochs using the constant init_learning_rate.
    max_epoch: total number of epochs in training
    input_size: size of the sliding window / one training data point
    batch_size: number of data points to use in one mini-batch.

The LSTM model has num_layers stacked LSTM layer(s) and each layer contains lstm_size number of LSTM cells. Then a dropout mask with keep probability keep_prob is applied to the output of every LSTM cell. The goal of dropout is to remove the potential strong dependency on one dimension so as to prevent overfitting.

The training requires max_epoch epochs in total; an epoch is a single full pass of all the training data points. In one epoch, the training data points are split into mini-batches of size batch_size. We send one mini-batch to the model for one BPTT learning. The learning rate is set to init_learning_rate during the first init_epoch epochs and then decay by × learning_rate_decay during every succeeding epoch.

Configuration is wrapped in one object for easy tracking and passing.

        input_size=1
        num_steps=30
        lstm_size=128
        num_layers=1
        keep_prob=0.8
        batch_size = 64
        init_learning_rate = 0.001
        learning_rate_decay = 0.99
        init_epoch = 5
        max_epoch = 50
        
## Good LSTM Pytorch setup article : https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/
## GRU vs LSTM with Pytorch : https://blog.floydhub.com/gru-with-pytorch/