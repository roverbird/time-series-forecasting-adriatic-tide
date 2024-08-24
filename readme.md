# Forecasting time series with SARIMA: Adriatic Tide Level in Koper, Slovenia

Welcome to the tide forecasting project for Koper, Slovenia. This repository contains code and documentation for forecasting tide levels using machine learning techniques on hourly time series data. Specifically, we'll explore and apply suitable models to predict future tide levels based on historical data from the Koper tide station. Data comes from here: [Postaja Koper - Kapitanija - Jadransko Morje](http://rte.arso.gov.si/vode/podatki/amp/H9350_t_1.html).

We will use this model for creating a forecast: Seasonal Autoregressive Integrated Moving Average (SARIMA)

![Flood in Izola near Koper](https://github.com/roverbird/time-series-forecasting-adriatic-tide/blob/main/seasonTide/data/floodIzola.png)

_A recent flood in Izola near Koper_

## Overview

Tide forecasting is used in various applications, including maritime navigation, coastal management, and environmental monitoring. In this project, we'll use historical tide level data to develop a predictive model that can forecast future tide levels. We'll leverage time series analysis and machine learning techniques to achieve accurate predictions.

There is a very informative article, [Seasonality Analysis and Forecast in Time Series](https://medium.com/swlh/seasonality-analysis-and-forecast-in-time-series-b8fbba820327) on making predictions with time series data using SARIMA, but that article lacks usable code and dataset. This repo closly follows instructions from Ayşenur Özen and provides python scripts and data for the task.

## Data Description

- **Source**: Postaja Koper - Kapitanija - Jadransko Morje, Koper, Slovenia
- **Frequency**: Hourly
- **Duration**: 30 days
- **Format**: CSV file with columns:
  - `date`: Date and time of the observation
  - `tide`: Recorded tide level in mm at the corresponding time

## Project Workflow

1. **Data Preparation**:
   - Load and preprocess the data.
   - Ensure chronological order of the time series.

2. **Exploratory Data Analysis (EDA)**:
   - Visualize the data to understand trends, seasonality, and any anomalies.

![Plot example 1](https://github.com/roverbird/time-series-forecasting-adriatic-tide/blob/main/seasonTide/data/scatter_plot_x_freq.png)

_Plot of Koper's 30 days of tide data_

3. **Seasonal Decomposition**:
   - Decompose the time series into trend, seasonal, and residual components to better understand the underlying patterns.

![Plot example 2](https://github.com/roverbird/time-series-forecasting-adriatic-tide/blob/main/seasonTide/data/seasonal_decomposition_x_freq.png)

_Time series divided into trend, seasonal and residual components_

4. **Model Selection and Training**:
   - Use the SARIMA (Seasonal AutoRegressive Integrated Moving Average) model to forecast tide levels.
   - Apply `auto_arima` from the `pmdarima` library to identify the best SARIMA model configuration.

5. **Evaluation**:
   - Split the data into training and testing sets.
   - Fit the model on the training data and evaluate its performance on the test data using metrics like Mean Squared Error (MSE).

![Plot example 3](https://github.com/roverbird/time-series-forecasting-adriatic-tide/blob/main/seasonTide/data/sarima_forecast.png)

_Predicting 12 hours of Adriatic tide level vs real observations_

6. **Visualization**:
   - Create plots to visualize the training data, test data, and forecast results.
   - Save the plots as PNG files for documentation and analysis.

## Checking for Stationarity in Tide Data

Before applying the SARIMA model, it's required to ensure that the time series data is stationary. A stationary time series has constant mean and variance over time, which is a key assumption for many time series forecasting models, including SARIMA.

In the `tideADF.py` script, we used the Augmented Dickey-Fuller (ADF) test to check for stationarity in the tide level data. Here’s a brief summary of the results:

- **ADF Statistic**: -18.0027
- **p-value**: 2.7310e-30
- **Number of Lags Used**: 24
- **Number of Observations Used**: 1414
- **Critical Values**:
  - **1%**: -3.435
  - **5%**: -2.864
  - **10%**: -2.568

**Interpretation**:
- **ADF Statistic**: The test statistic is well below the critical values at 1%, 5%, and 10% levels.
- **p-value**: The p-value is significantly less than 0.05.

Since the p-value is much smaller than the typical significance level (0.05), and the ADF statistic is lower than the critical values, we reject the null hypothesis. This indicates that the tide data series is likely stationary.

## Code Walkthrough

1. **Data Loading and Preparation**:
   ```python
   import pandas as pd

   # Load the data
   df = pd.read_csv('~/seasonTide/data/input.csv')

   # Convert 'date' to datetime and sort the data
   df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y-%H%M')
   df = df.sort_values(by='date')
   df.set_index('date', inplace=True)
   ```

2. **Seasonal Decomposition**:
   ```python
   import statsmodels.api as sm

   # Perform seasonal decomposition
   result = sm.tsa.seasonal_decompose(df['tide'], model='additive', period=24)
   ```

3. **Model Training and Forecasting**:
   ```python
   import pmdarima as pm

   # Fit SARIMA model
   sarima_model = pm.auto_arima(
       df['tide'][:-12], 
       seasonal=True, 
       m=24,
       trace=True, 
       error_action='ignore',  
       suppress_warnings=True,  
       stepwise=True
   )

   # Forecast the next 12 observations
   forecast = sarima_model.predict(n_periods=12)
   ```

4. **Visualization**:
   ```python
   import plotly.graph_objs as go

   # Create the plot
   fig = go.Figure()

   # Add traces
   fig.add_trace(go.Scatter(x=df.index[:-12], y=df['tide'][:-12], mode='lines', name='Training Data'))
   fig.add_trace(go.Scatter(x=df.index[-12:], y=df['tide'][-12:], mode='lines', name='Test Data', line=dict(color='orange')))
   fig.add_trace(go.Scatter(x=df.index[-12:], y=forecast, mode='lines', name='Forecast', line=dict(color='green')))

   # Update layout and save
   fig.update_layout(title='SARIMA Model - Train/Test Split and Forecast', xaxis_title='Date', yaxis_title='Tide', template='plotly_white')
   fig.write_image('~/seasonTide/data/sarima_forecast.png')
   ```

## Results

- **Mean Squared Error (MSE)**: The model's accuracy was evaluated using the Mean Squared Error metric on the test data.
- **Forecast Visualization**: The forecasted tide levels and their comparison with actual observations are visualized in the saved PNG file.

## Conclusion

This project demonstrates how machine learning techniques can be applied to time series data for tide level forecasting. The SARIMA model provides a robust approach to capturing both seasonal and trend components in the tide data. Further refinements and exploration of advanced models could improve the accuracy of long-term forecasts.

## Future Work

- Explore additional models (e.g., Prophet, LSTM) for better performance.
- Incorporate external factors such as weather data for improved accuracy.
- Extend the forecast horizon and evaluate the model's robustness over longer periods.

## Dependencies

Ensure you have the following Python packages installed:

- `pandas`
- `pmdarima`
- `plotly`
- `statsmodels`
- `scikit-learn`

Install them using pip if needed:

```bash
pip install pandas pmdarima plotly statsmodels scikit-learn
```

## Repository Structure

Here’s a brief comment on the contents of the repository:

- **`/bin`**: Contains scripts for various stages of the analysis.
  - **`tideADF.py`**: Performs the Augmented Dickey-Fuller (ADF) test to check for stationarity in the tide data.
  - **`tideDecomp.py`**: Decomposes the tide time series into trend, seasonal, and residual components.
  - **`tideForecast.py`**: Fits the SARIMA model to the tide data and generates forecasts.
  - **`tideTest.py`**: Handles train/test split and evaluates the SARIMA model's performance.

- **`/data`**: Stores data files and output images.
  - **`input.csv`**: Raw tide level data in CSV format.
  - **`sarima_forecast.png`**: Visualization of SARIMA model forecasts compared to actual data.
  - **`scatter_plot_x_freq.png`**: Scatter plot of tide data over time.
  - **`seasonal_decomposition_x_freq.png`**: Visualization of the seasonal decomposition of tide data.

- **`readme.md`**: Provides an overview of the project, including data description, methodology, and results.

/seasonTide/
├── bin
│   ├── tideAFD.py
│   ├── tideDecomp.py
│   ├── tideForecast.py
│   └── tideTest.py
├── data
│   ├── input.csv
│   ├── sarima_forecast.png
│   ├── scatter_plot_x_freq.png
│   └── seasonal_decomposition_x_freq.png
└── readme.md

![Adriatic Sea](https://github.com/roverbird/time-series-forecasting-adriatic-tide/blob/main/seasonTide/data/JadranskoMorje.png)

_Rough Adriatic sea near Koper during the flood season_

## Context and Importance of Tide Forecasting

The recent devastating floods in Venice, often referred to as "acqua alta" (high water), highlight the critical need for accurate tide forecasting systems. Venice, known for its intricate network of canals and historic architecture, is particularly vulnerable to sea level rise and extreme tidal events. Such floods not only disrupt daily life but also pose significant risks to cultural heritage and infrastructure.

Similarly, the vicinity of Koper, including the nearby town of Izola, faces its own challenges with frequent flooding. Izola, located just a few kilometers from Koper, has recently experienced regular inundations due to rising tide levels. This highlights the importance of having a robust prediction system for tide levels in the region. Accurate forecasting can provide advance warning, enabling better preparation and mitigation strategies for both Koper and its neighboring communities. Such systems help in minimizing the impacts of flooding on infrastructure, local economies, and daily activities.

In this project, we show how one can develop and apply machine learning techniques to forecast tide levels, which could be a crucial step towards enhancing flood resilience in the region. Is there a public system to predic floods in Kopen, Izola and Piran in place, similar to Venice? Did anyone bother?

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

