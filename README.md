# This project is an exploratory project to learn about RNN, and apply it on real market data

## The application goal is to predict the next business day closing exchange rate between EUR and CAD
## The learning goal is to learn and practice on :
- Pytorch
- RNN (LSTM, GRU...)
- Graph construction
- Web app integration and result visualization (Plotly Dash or Streamlit ?)


### Things to try :
- Fixed rolling windows
- LSTM
- GRU
- Data augmentation with very small noise 


## How to use it

Paragraph under construction



###Steps : 
- Data Gathering
    - Web Scrapping with BeautifulSoup
- Data Preparation
    - Data normalization
        - We change the raw exchange rate value to a delta from previous day. To avoid having to bound 
        min/max value and to avoid other tricks of normalization. This should be enough.
    - Data split
        - Split should not be random because of periodicity. Split should cut the train/test on a specific date
- Model Construction
    - Definition
    - Graph definition
- Model Train
    - Evaluation and boards
    - Gather results


To do list :
* Add other date info for information (holiday, christmas, or even raw day/month) ?
* Try a longer full LSTM sequence ? (not only the N last to guess the next)