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

### LSTM best explaination : https://colah.github.io/posts/2015-08-Understanding-LSTMs/
### https://lilianweng.github.io/lil-log/2017/07/08/predict-stock-prices-using-RNN-part-1.html#define-graph        
### Good LSTM Pytorch setup article : https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/
### GRU vs LSTM with Pytorch : https://blog.floydhub.com/gru-with-pytorch/

## Loss choice :
- MAE : 
    - Useful when measuring prediction errors in the same unit as the original series
    - Quite robust to ourliers
    - Tell you how big of an error can you expect from the forecast on average
    - Easy to interpret
    - If the data is homogeneous, use this error measure to compare between different models.
    - MAE is not unique and hence might seem to show schizophrenic behaviour ?
- MedAE :
    - Like MAE
    - Also allow missing values
    - Good to trim extreme values
    - Reduce bias
- **MSE** : 
    - Average of the square of the forecast error
    - The effect is that larger errors have more weight on the score
    - Most used to evaluate and find model
    - Hard to use the raw value
    - Outliers will have a huge effect on the resulting error
- **RMSE** : 
    - Root of MSE before dividing it with sample size
- MAPE : 
    - Gives a good idea of the relative error
    - When the forecast series can have small denominators (chance of divise by 0)
    - Heavy penalty on negative errors where y<yhat (Not suitable in our case !!!)
    - Tend to select a method whose forecasts are too low
- sMAPE :
    - Symmetric MAPE
    - Has both a lower bound and an upper bound
    - Not symmetric between under and over forecasting (Bad for our case)