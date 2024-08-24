import pandas as pd
import statsmodels.api as sm
import plotly.graph_objs as go
import plotly.subplots as sp

# Load the data from the CSV file
df = pd.read_csv('/seasonTide/data/input.csv')

# Ensure the 'date' column is treated as a date
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y-%H%M')

# Set 'date' as the index
df.set_index('date', inplace=True)

# Perform seasonal decomposition using an additive model
result = sm.tsa.seasonal_decompose(df['tide'], model='additive', period=24)

# Extract components
trend = result.trend
seasonal = result.seasonal
residual = result.resid
observed = result.observed

# Create subplots for the decomposition
fig = sp.make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.1, 
                       subplot_titles=["Observed", "Trend", "Seasonal", "Residual"])

# Plot the observed data
fig.add_trace(go.Scatter(x=df.index, y=observed, mode='lines', name='Observed'), row=1, col=1)

# Plot the trend component
fig.add_trace(go.Scatter(x=df.index, y=trend, mode='lines', name='Trend'), row=2, col=1)

# Plot the seasonal component
fig.add_trace(go.Scatter(x=df.index, y=seasonal, mode='lines', name='Seasonal'), row=3, col=1)

# Plot the residual component
fig.add_trace(go.Scatter(x=df.index, y=residual, mode='lines', name='Residual'), row=4, col=1)

# Update the layout
fig.update_layout(height=800, width=1200, title_text="Seasonal Decomposition of Tide", 
                  template='plotly_white')

# Save the plot as a PNG file
fig.write_image('/seasonTide/data/seasonal_decomposition_x_freq.png')

