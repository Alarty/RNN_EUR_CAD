# This project is an exploratory project to learn about RNN, and apply it on real market data

## The application goal is to predict the next business day closing exchange rate between EUR and CAD
## The learning goal is to learn and practice on :
- Pytorch
- RNN (LSTM, GRU...)
- Web app integration and result visualization (Plotly Dash or Streamlit ?)

###Steps of the scripts: 
- Data Gathering
    - Web Scrapping with BeautifulSoup or a CSV file
- Data Preparation
    - Data normalization
        - We change the raw exchange rate value to a delta from previous day. To avoid having to bound 
        min/max value and to avoid other tricks of normalization. This should be enough.
    - Data split
        - Split should not be random because of periodicity. Split should cut the train/test on a specific date
- Model Construction
    - Definition
        - Try GRU, LSTM and ARIMA to compare these models and learn the pros/cons
    - Parameters
        - Use Adam optimizer and MSE loss
        - 1 hidden layer of 256 neurons to start, may evolve
        - Learning rate to tweak according to loss curves
- Model Train
    - Evaluation and boards
        - Tensorboard setup to monitor training and results
    - Gather results

### To do list (sorted by priority) :
* use as input open/close/min/max of previous days instead of only close ?
* Verify LSTM implement issue ?
* StandardScaler on train set ? to 
* Data Augment with noise
* Tweak model parameters
* Save and reload models to gain time
* Add other date info as features post-RNN (holiday, christmas, or even raw day/month) ?
* Try ARIMA and facebook prophet

### Launch Tensorboard on Windows
> .\venv\Scripts\python.exe .\venv\Lib\site-packages\tensorboard\main.py --logdir=runs

