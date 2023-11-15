import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Assuming 'data' is your time series data
data = pd.read_excel('cis presentation.xlsx', sheet_name='Sheet3')

# Fit the model
model = ARIMA(data, order=(5,5,5))
model_fit = model.fit()

# Number of future points to predict
n_forecast = 20

# Predict future data
forecast = model_fit.forecast(steps=n_forecast)

print('Forecast: ', forecast)

# Create a new figure
plt.figure()

# Plot the current data
plt.plot(data.index, data, label='Current')

# Generate future indices
future_index = range(data.index[-1] + 1, data.index[-1] + n_forecast + 1)
plt.xlabel('Time(days)')
plt.ylabel('Utilization')
# Plot the future data
plt.plot(future_index, forecast, label='Future')

# Add a legend
plt.legend()

# Show the plot
plt.show()