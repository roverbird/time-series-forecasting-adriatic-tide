import pandas as pd
import pmdarima as pm
import plotly.graph_objs as go
import plotly.io as pio
from sklearn.metrics import mean_squared_error

# Load the data from the CSV file
df = pd.read_csv('/seasonTide/data/input.csv')

# Ensure the 'date' column is treated as a date
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y-%H%M')

# Reverse the DataFrame to have the oldest data first
df = df.sort_values(by='date')

# Set 'date' as the index
df.set_index('date', inplace=True)

# Define the train/test split
train_data = df['tide'][:-12]
test_data = df['tide'][-12:]

# Fit the SARIMA model to the training data
sarima_model = pm.auto_arima(
    train_data, 
    seasonal=True, 
    m=24,  # Adjust based on seasonality (24 hours for daily seasonality) # one line on input data is one hour of observations
    start_p=5, start_q=0, start_P=0, start_Q=0,
    max_p=5, max_q=0, max_P=0, max_Q=0,
    d=0, D=0,  # No differencing, as identified earlier
    trace=True, 
    error_action='ignore',  
    suppress_warnings=True,  
    stepwise=True
)

# Forecast the next 12 observations (test set)
forecast = sarima_model.predict(n_periods=12)

# Evaluate the forecast
mse = mean_squared_error(test_data, forecast)
print(f"Mean Squared Error: {mse}")

# Create the plot
fig = go.Figure()

# Add the training data
fig.add_trace(go.Scatter(
    x=train_data.index, 
    y=train_data, 
    mode='lines', 
    name='Training Data'
))

# Add the test data
fig.add_trace(go.Scatter(
    x=test_data.index, 
    y=test_data, 
    mode='lines', 
    name='Test Data', 
    line=dict(color='orange')
))

# Add the forecast data
fig.add_trace(go.Scatter(
    x=test_data.index, 
    y=forecast, 
    mode='lines', 
    name='Forecast', 
    line=dict(color='green')
))

# Update layout
fig.update_layout(
    title='SARIMA Model - Train/Test Split and Forecast',
    xaxis_title='Date',
    yaxis_title='Tide',
    template='plotly_white',
    width=1500,  # Set the width to make the plot wider
    height=500   # Adjust the height if needed
)

# Save the plot as a PNG file
fig.write_image('/seasonTide/data/sarima_forecast.png')

# Display the plot (optional)
# fig.show()

