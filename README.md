# Bitcoin Price Forecasting with LSTM and Backtesting

Final project exploring whether an LSTM-based time-series model, combined with technical indicators, can produce useful Bitcoin price forecasts and whether those forecasts translate into a profitable trading strategy.

This project goes beyond pure prediction by evaluating the model in a more practical setting through walk-forward backtesting, trading simulation, and directional classification.

---

## Overview

Bitcoin is highly volatile, which makes forecasting both interesting and difficult.  
In this project, I used historical BTC/USD price data from the CoinGecko API and engineered a set of technical indicators to train recurrent neural networks for forecasting and directional prediction.

The main question was not just:

> *Can an LSTM fit Bitcoin prices reasonably well?*

but also:

> *Do those predictions actually help in a trading scenario?*

---

## What this project includes

- historical Bitcoin price retrieval from the CoinGecko API
- feature engineering with technical indicators
- LSTM regression for price prediction
- walk-forward backtesting
- trading simulation with transaction fees
- threshold-based trading logic
- directional classification with LSTM
- logistic regression baseline
- a simple news-fetching component for qualitative market context

---

## Features used

The model was trained using Bitcoin price history together with the following technical indicators:

- **SMA 20**
- **SMA 50**
- **RSI**
- **MACD**
- **Bollinger Bands**
  - middle band
  - upper band
  - lower band

All features were scaled using `MinMaxScaler` before training.

---

## Model setup

### Regression model
The main forecasting model is a stacked LSTM network built with TensorFlow/Keras:

- 2 LSTM layers
- dropout regularization
- dense output layer
- trained for 10 epochs
- 80/20 train-test split

### Directional classification model
A second LSTM model was trained to predict whether the next price move would be up or down.

### Baseline model
A logistic regression classifier was also used as a simple baseline for directional prediction.

---

## Results

### 1. Price prediction (regression)

The LSTM regression model fit the held-out test set reasonably well:

- **RMSE:** 780.57
- **MAE:** 604.78
- **R²:** 0.9191

These results suggest that the model captured the general structure of Bitcoin price movements fairly well over the evaluation period.

---

### 2. Directional accuracy

Even though the regression fit looked strong, the model was much weaker at predicting short-term direction:

- **Directional accuracy from regression predictions:** 0.5036
- **Threshold-based directional accuracy:** 0.5281

This is only slightly above chance and shows an important limitation:  
good-looking regression metrics do not necessarily imply useful directional signals.

---

### 3. Walk-forward backtesting and trading simulation

To test whether predictions were useful in practice, I ran a walk-forward backtest and simulated trading using model-generated signals.

#### Basic LSTM trading strategy
- **Final value (LSTM strategy):** \$97,773.51
- **Final value (buy & hold):** \$121,199.88
- **Outperformance:** **-\$23,426.38**

#### Threshold-based strategy
- **Final value (threshold strategy):** \$108,390.95
- **Final value (buy & hold):** \$121,199.88
- **Outperformance:** **-\$12,808.93**

### Key takeaway
The model produced forecasts that looked reasonable on standard regression metrics, but those forecasts did **not** translate into a trading strategy that beat buy-and-hold over the same period.

This is one of the most important conclusions of the project.

---

### 4. Directional classification

The LSTM classifier achieved:

- **Accuracy:** 0.5333

However, the confusion matrix showed that the model predicted only one class:

```text
[[  0 196]
 [  0 224]]
````

The classification report confirms that this result is not genuinely strong despite the headline accuracy.

This was useful because it highlighted how misleading raw accuracy can be in time-series direction prediction.

---

### 5. Logistic regression baseline

The logistic regression baseline achieved:

* **Accuracy:** 0.5308

This baseline performed similarly to the LSTM classifier, reinforcing the conclusion that directional prediction remained difficult with the available features and data.

---

## Main conclusion

This project shows a useful distinction between:

* **predicting price levels reasonably well**
* and **producing actionable trading signals**

The LSTM model achieved a strong **R² of 0.9191**, but directional prediction stayed close to chance, and both simulated trading strategies underperformed a simple buy-and-hold benchmark.

In other words:

> a model can appear successful on standard forecasting metrics while still failing to generate practical trading value.

That is the central lesson of this project.

---

## Limitations

A few important limitations affected the results:

* the project used a relatively short historical window
* the market regime during the sampled period was unusually volatile
* trading performance is sensitive to fees and frequent switching
* technical indicators alone may not capture major external drivers
* news, macro events, regulation, and market sentiment were not integrated directly into the predictive model

The notebook also includes a simple news-fetching step to provide qualitative market context, reflecting the idea that human judgment may still be necessary when interpreting model outputs in financial markets.

---

## Repository contents

* `final_project.ipynb` — full notebook with data collection, feature engineering, modeling, backtesting, and outputs
---

## Tech stack

* Python
* pandas
* NumPy
* matplotlib
* scikit-learn
* TensorFlow / Keras
* ta
* requests

---

## How to run

1. Create and activate a virtual environment
2. Install the required packages
3. Open the notebook and run all cells

Example:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

pip install pandas numpy matplotlib scikit-learn tensorflow ta requests notebook
jupyter notebook
```

Then open:

```text
Final_Project.ipynb
```

---

## Future improvements

Some directions that would make this project stronger:

* use a longer historical horizon
* test higher-frequency data
* add external signals such as sentiment or macro indicators
* compare against more baselines
* use better evaluation for directional prediction
* refactor the notebook into a cleaner pipeline with separate modules/scripts

---

## Author

Final course project by **Cameron Caputa** and **Giacomo Sforza**.
