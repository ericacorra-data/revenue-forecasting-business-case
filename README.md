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

```python

```
```python

```
```python

```
```python

```
```python

```
```python

```
```python

```
```python

```
```python

``````python

```















