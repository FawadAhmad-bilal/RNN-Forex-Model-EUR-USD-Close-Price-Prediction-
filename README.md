# RNN Forex Model (EUR/USD Close Price Prediction)

A simple RNN-based time series model that predicts the closing price of EUR/USD using historical OHLC (Open, High, Low, Close) data.

## Overview

This project uses a `SimpleRNN`-based neural network to forecast the next-step closing price of the EUR/USD currency pair, using a sliding window of the past 20 days of price data as input.

## Data

- **Source:** [Yahoo Finance](https://finance.yahoo.com/) via the `yfinance` library
- **Ticker:** `EURUSD=X`
- **Date range:** 2010-01-01 to present
- **Features used:** Open, High, Low (Volume dropped)
- **Target:** Close price

## Preprocessing

- Train/test split: 80/20 (chronological, not shuffled — important for time series)
- Feature scaling: `MinMaxScaler` (fit on train, applied to test)
- Sequence creation: sliding windows of 20 time steps to predict the next day's close

## Model Architecture

```
SimpleRNN(16, return_sequences=True)
SimpleRNN(16, return_sequences=False)
Dense(40, activation='relu')
Dense(1, activation='linear')
```

- **Loss:** Mean Squared Error (MSE)
- **Optimizer:** Adam
- **Metric:** R² Score
- **Epochs:** 25

## Results

| Metric | Value |
|--------|-------|
| R² Score | *(add your value here)* |
| MAE | *(add your value here)* |
| MSE | *(add your value here)* |

### Training Curves

![R2 Score and Loss Curves](images/training_curves.png)

*Left: R² score over epochs (train vs validation). Right: MSE loss over epochs (train vs validation).*

## Known Limitations

- `SimpleRNN` is prone to vanishing gradients on longer sequences — an LSTM or GRU is likely to perform better and is a planned next step.
- The target variable (`Close`) is not scaled, which can slow convergence.
- No `EarlyStopping` or `ModelCheckpoint` is currently used, so the model trains for a fixed number of epochs regardless of overfitting.
- Forex price prediction from OHLC data alone is inherently noisy; this project is for learning purposes, not live trading.

## How to Run

```bash
pip install yfinance scikit-learn tensorflow matplotlib
python rnn_forex_model.py
```

## Next Steps

- Replace `SimpleRNN` with `LSTM`/`GRU`
- Scale the target variable and inverse-transform predictions before evaluation
- Add `EarlyStopping` and `ModelCheckpoint`
- Fix metric argument order (`r2_score(y_true, y_pred)`, not `r2_score(y_pred, y_true)`)
