import pandas as pd
import pmdarima as pm
import matplotlib.pyplot as plt

# Load the data from the CSV file
df = pd.read_csv('/home/sanka/seasonTide/data/input.csv')

# Ensure the 'date' column is treated as a date
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y-%H%M')
df = df.sort_values(by='date')


# Set 'date' as the index
df.set_index('date', inplace=True)

# Fit the SARIMA model using auto_arima
sarima_model = pm.auto_arima(
    df['tide'], 
    seasonal=True, 
    m=24,  # Frequency of the seasonality (e.g., 6 for daily data with weekly seasonality)
    trace=True,  # Set to True to see the output in the console
    error_action='ignore',  # Ignore if any model fails to fit
    suppress_warnings=True,  # Suppress warnings
    stepwise=True  # Use stepwise approach to search for the best parameters
)

# Summary of the SARIMA model
print(sarima_model.summary())

# Plot diagnostics
sarima_model.plot_diagnostics(figsize=(12, 8))
plt.show()
