# Revenue Forecasting for Strategic Business Planning

### Supporting Strategic Business Planning through Revenue Forecasting

**Author:** Erica Corradini

---

### Introduction

Accurate revenue forecasting is essential for strategic business planning.
In this project, historical sales data stored in a relational SQLite database are used to predict the next 12 months of revenue through statistical forecasting models and Monte Carlo simulation.
The objective is to compare different forecasting techniques, evaluate their predictive performance, and provide actionable insights to support business decision-making.

### Why this project?

Many companies rely on accurate revenue forecasts to support strategic planning.
This project demonstrates how statistical forecasting models and scenario analysis can be combined to improve business decision-making.

> **Used Tools**
>
> Python • Pandas • NumPy • Statsmodels • Scikit-learn • SQLite • Matplotlib • Seaborn
>
>  ### Key Results
>
> **Best Forecasting Model:**  Holt-Winters Exponential Smoothing
>
> **Expected Revenue:** €437.07 M
>
> **Probability of exceeding €400M:** 100%
>
> **Forecast Accuracy:**  Lowest RMSE among all evaluated models
>
> **Risk Assessment:** 1,000 Monte Carlo simulations
>
**Business Insight:** Seasonal forecasting models (Holt-Winters and SARIMA) consistently outperformed non-seasonal approaches, highlighting seasonality as the primary driver of revenue dynamics. These results indicate that incorporating seasonal effects is essential for reliable business forecasting and strategic planning.
## Table of Contents
1. [Business Problem](#business-problem)

2. [Business Questions](#business-questions)

3. [Dataset Overview](#dataset-overview)

4. [Data Preparation](#importing-and-cleaning-the-data)

5. [Exploratory Data Analysis](#exploratory-data-analysis)

6. [Forecasting Models](#forecasting-models)

7. [Model Evaluation](#model-evaluation)

8. [Monte Carlo Simulation](#monte-carlo-simulation)

9. [Business Conclusions](#business-conclusions)

10. [Final takeaway](#final-takeaway)

## Business Problem

Reliable revenue forecasts are essential for effective business planning.
This project develops a forecasting framework capable of estimating future monthly revenue while quantifying forecast uncertainty.
The results support budgeting, production planning, inventory management, and strategic decision-making.
## Business Questions

This project aims to answer the following business questions:

- How much revenue can the company expect next year?
- Which forecasting model performs best?
- Does the revenue exhibit seasonality?
- How reliable are the forecasts?
- What is the probability of exceeding the annual revenue target? 
- How can forecasting support business planning?

## Dataset Overview

The analysis is based on a relational SQLite database simulating approximately ten years of manufacturing operations.
Monthly revenue is calculated as:
Revenue = Quantity × Unit Price

The resulting time series is used to train and evaluate multiple forecasting models.

| Dataset | Description |
|----------|-------------|
| Source | SQLite Database |
| Industry | Manufacturing |
| Period | 2015-2025 |
| Frequency | Monthly |
| Target Variable | Revenue (€) |

## Importing and cleaning the data

# Import Required Libraries
import sqlite3

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)
from math import sqrt
from IPython.core.display import HTML as Center
```python
# Import Required Libraries
import sqlite3

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)
from math import sqrt
from IPython.core.display import HTML as Center
from warnings import filterwarnings

filterwarnings("ignore")
```
```python
#Centering the plots
Center(""" <style>
.jp-RenderedImage {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
}
</style> """)
```
```python
# Seaborn visualization settings:
sns.set_theme(
    style="darkgrid",      
    context="notebook",        
    palette="deep"
)
```
```python
#Connect to the SQLite database used for revenue forecasting analysis
conn = sqlite3.connect("data/ceramica_pro17.db")
```
```text
Dataset loaded
```
```python
vendite.info()
```
###  Cleaning data
Cleaning data is crucial because it directly impacts the accuracy, consistency, and reliability of any analysis, model, or decision-making process.

```python
# Convert dates to datetime format
vendite["data"] = pd.to_datetime(vendite["data"])
# Calculate revenue for each sale as quantity × unit price
vendite["fatturato"] = vendite["quantita"] * vendite["prezzo"]
vendite.head()
```
| Date | Customer | Product | Quantity | Unit Price (€) | Revenue (€) |
|:------------|---------:|--------:|---------:|---------------:|------------:|
| 2023-09-16 | 74 | 5 | 2,409 | 20.47 | 49,312.23 |
| 2023-10-30 | 70 | 2 | 2,781 | 12.88 | 35,819.28 |
| 2016-09-01 | 96 | 5 | 3,923 | 20.27 | 79,519.21 |
| 2019-05-19 | 20 | 4 | 1,516 | 35.45 | 53,742.20 |
| 2019-05-19 | 106 | 4 | 1,516 | 36.18 | 54,848.88 |
### Dealing with null values
This refers to handling the missing or absent data in a dataset. In many datasets, especially large or real-world ones, it’s common to encounter missing values. 
```python
vendite.isnull().sum()
```
```text
data         0
cliente      0
prodotto     0
quantita     0
prezzo       0
fatturato    0
dtype: int64
```
✔ No missing values were detected.
### Dealing with duplicates
This is the process of identifying and managing duplicate records in a dataset. Duplicates can occur when the same data is entered multiple times.
```text
Duplicate rows: 18
```
No duplicate records were removed because duplicate observations represent less than 0.03% of the dataset and therefore have a negligible impact on the analysis.
## Exploratory Data Analysis
---
```python
# Create a time-series structure by using dates as the index and ordering records chronologically
vendite = vendite.set_index("data")
vendite = vendite.sort_index()
monthly_revenue = vendite["fatturato"].resample("ME").sum()
# Resample transaction data by month-end and calculate total monthly revenue for time-series analysis
monthly_revenue = vendite["fatturato"].resample("ME").sum()
```
```python
plt.figure(figsize=(12,3))
plt.plot(monthly_revenue.index, monthly_revenue / 1000000.0, linewidth=2)
plt.title("Revenue by month (2015-2025)", fontsize=16, fontweight="bold")
plt.xlabel("Year",fontsize=14, fontweight="bold")
plt.ylabel("Revenue (M€)",fontsize=14,fontweight="bold")
plt.grid(True)
sns.despine()
plt.show()
```
![Revenue Forecast](revenue_files/revenue_by_month.png)
## Forecasting Models
#### ARIMA
The optimal ARIMA configuration was selected through a systematic comparison of multiple candidate models using AIC, BIC and out-of-sample RMSE.

```python
# Fit an ARIMA(2,1,1) model to the monthly revenue series
model_ARIMA = ARIMA (monthly_revenue, order = (2,1,1))
# Estimate model parameters
fit_ARIMA = model_ARIMA.fit()
# Generate a 12-month revenue forecast
forecast_ARIMA_res = fit_ARIMA.get_forecast(steps=12)
# Extract forecasted values
forecast_ARIMA = forecast_ARIMA_res.predicted_mean
# Calculate 95% confidence intervals
conf_int_ARIMA = forecast_ARIMA_res.conf_int(alpha=0.05)
```
The ARIMA model forecasts gradually converge toward a stable long-term value of approximately €33.82M per month. This behavior is expected, as ARIMA captures trend and autocorrelation patterns but does not explicitly model seasonality. As a result, long-term forecasts tend to revert toward an equilibrium level rather than reproducing recurring seasonal fluctuations. The forecast for January 2025 is associated with a particularly wide 95% confidence interval (€6.82M–€49.90M), indicating substantial uncertainty in the prediction. 
- ARIMA captures the long-term trend but fails to reproduce the strong annual seasonality observed in the historical revenue series. Consequently, it serves as a baseline model for comparison with seasonal forecasting approaches.
#### SARIMA
Three SARIMA configurations were evaluated to identify the most accurate seasonal forecasting model.
SARIMA(1,1,2)(1,1,1,12) achieved the lowest forecasting error, confirming that explicitly modeling annual seasonality substantially improves predictive performance compared with the baseline ARIMA model.
These results suggest that seasonality is one of the main drivers of revenue dynamics and should be incorporated into business forecasting models.
```python
# Fit a seasonal ARIMA model (SARIMA) to capture both trend and annual seasonality in the revenue series
model_SARIMA = SARIMAX(monthly_revenue, order=(1,1,2), seasonal_order =(1,1,1,12))
# Estimate model parameters
fit_SARIMA = model_SARIMA.fit()
# Generate a 12-month forecast
forecast_SARIMA_res = fit_SARIMA.get_forecast(steps=12)
# Extract forecasted values
forecast_SARIMA = forecast_SARIMA_res.predicted_mean
# Calculate 95% confidence intervals
conf_int_SARIMA = forecast_SARIMA_res.conf_int(alpha=0.05)
```
Unlike the ARIMA model, the SARIMA forecasts exhibit clear seasonal fluctuations throughout the forecast horizon, closely reflecting the recurring patterns observed in the historical revenue series. Forecasted revenue varies significantly across months, with peaks reaching approximately €47.19M in May and lower values around €10.84M in August, indicating that the model successfully captures the annual seasonality present in the data. The 95% confidence intervals remain relatively wide, reflecting the inherent uncertainty of long-term forecasting. 
- SARIMA successfully captures the annual seasonal pattern, producing forecasts that closely follow historical revenue dynamics and substantially improve predictive accuracy over ARIMA.

```python
plt.figure(figsize=(12,3))
# Storico
plt.plot(monthly_revenue.index, monthly_revenue.values / 1_000_000, marker = 'o', label = "Storico")

# Forecast
plt.plot(forecast_ARIMA.index, forecast_ARIMA.values / 1_000_000, marker = 'x', linestyle = '--', color='red', label = "Forecast ARIMA")
plt.fill_between(forecast_ARIMA.index, conf_int_ARIMA.iloc[:,0].values / 1_000_000, conf_int_ARIMA.iloc[:,1].values / 1_000_000, 
                 color = 'pink', alpha =0.3, label ='95% CI ARIMA')

plt.plot(forecast_SARIMA.index, forecast_SARIMA.values / 1_000_000, marker = '^', linestyle = '--', color='blue', label = "Forecast SARIMA")
plt.fill_between(forecast_SARIMA.index, conf_int_SARIMA.iloc[:,0].values / 1_000_000, conf_int_SARIMA.iloc[:,1].values / 1_000_000,
                 color = 'green', alpha =0.3, label ='95% CI SARIMA')

plt.axvline(monthly_revenue.index[-1], color='grey', linestyle=':')
plt.title("Revenue Forecast for the Next 12 Months with Confidence Intervals",fontsize=16, fontweight="bold")
plt.xlabel("Year",fontsize=14, fontweight="bold")
plt.ylabel("Revenue (M€)",fontsize=14, fontweight="bold")
plt.grid(True)
plt.legend()
plt.show()
```
![Revenue Forecast](revenue_files/revenue_forecast.png)
```python
#### Holt-Winters

```python
# Apply Holt-Winters Exponential Smoothing to model trend and annual seasonality
modello_HW = ExponentialSmoothing(monthly_revenue, trend="add", seasonal="add", seasonal_periods=12)
# Estimate model parameters
fit_HW = modello_HW.fit()
# Generate a 12-month revenue forecast
forecast_esponenziale = fit_HW.forecast(12)
```
The Holt-Winters Exponential Smoothing model successfully captures both trend and seasonal patterns in the monthly revenue series and generates more realistic revenue projections by explicitly modeling seasonal effects. The similarity between Holt-Winters and SARIMA forecasts further confirms the presence of strong annual seasonality in the dataset.















