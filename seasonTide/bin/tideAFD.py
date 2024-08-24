import pandas as pd
from statsmodels.tsa.stattools import adfuller

# Load the data from the CSV file
df = pd.read_csv('/home/sanka/seasonTide/data/input.csv')

# Perform the Augmented Dickey-Fuller test
result = adfuller(df['tide'])

# Extract and print the results
adf_statistic = result[0]
p_value = result[1]
used_lag = result[2]
n_obs = result[3]
critical_values = result[4]
icbest = result[5]

print("ADF Statistic:", adf_statistic)
print("p-value:", p_value)
print("Number of Lags Used:", used_lag)
print("Number of Observations Used:", n_obs)
print("Critical Values:", critical_values)

# Interpretation
if p_value < 0.05:
    print("The series is likely stationary (reject null hypothesis).")
else:
    print("The series is likely non-stationary (fail to reject null hypothesis).")

