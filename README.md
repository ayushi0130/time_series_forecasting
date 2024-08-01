# Time Series Forecasting with Python

## Introduction

This project demonstrates how to perform time series forecasting in Python using Linear Regression, Naive Forecast, and Exponential Smoothing. We'll use historical gold prices data for illustration.

## Prerequisites

Ensure you have the necessary libraries installed:

- `numpy`
- `pandas`
- `seaborn`
- `matplotlib`
- `statsmodels`
- `scikit-learn`

Install them using pip if needed:

```sh
pip install numpy pandas seaborn matplotlib statsmodels scikit-learn
```

Step 1: Import Libraries and Load Data

Step 2: Preprocess Data

Step 3: Data Visualization

- Plotting Gold Prices Over Time
- Statistical Description of Data
- Box Plot by Year
- Month Plot
- Box Plot by Month

Step 4: Resample Data and Plot Yearly, Quarterly, and Decadal Means

- Yearly Mean
- Quarterly Mean
- Decadal Mean

Step 5: Calculate Coefficient of Variation (CV) by Year

Step 6: Split Data into Training and Testing Sets

Step 7: Linear Regression

Step 8: Naive Forecast

Step 9: Exponential Smoothing

### Result

- Gold price monthly(lineplot):
<img width="1236" alt="Screen Shot 2024-06-13 at 14 17 24" src="https://github.com/ayushi0130/time_series_forecasting/assets/128896031/0272279b-5704-40ed-b0da-028a993bd0b8">

- Gold price monthly(Boxplot):
<img width="1230" alt="Screen Shot 2024-06-13 at 14 16 30" src="https://github.com/ayushi0130/time_series_forecasting/assets/128896031/488fb560-8142-4fcc-907f-114fee47c057">

- Linear Regression:
<img width="1213" alt="Screen Shot 2024-06-13 at 22 27 06" src="https://github.com/ayushi0130/time_series_forecasting/assets/128896031/e6f1b1b8-5331-4577-89be-f1bb9e08a76c">

- Naive Forecast:
<img width="1192" alt="Screen Shot 2024-06-13 at 22 28 24" src="https://github.com/ayushi0130/time_series_forecasting/assets/128896031/66bcdebd-39ad-4d8f-b9c3-6053e1bf2d56">

- Exponentialsmoothing:
<img width="1273" alt="Screen Shot 2024-06-13 at 22 28 59" src="https://github.com/ayushi0130/time_series_forecasting/assets/128896031/13d5199d-a57c-46ee-aa1b-0d7ac7c8e393">

### Conclusion:

This project covered time series forecasting using Linear Regression, Naive Forecast, and Exponential Smoothing. Each method was evaluated using the Mean Absolute Percentage Error (MAPE) to compare their performances. The Exponential Smoothing model provided the final forecasts with confidence intervals.
