import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the dataset
file_path = 'D:/projects/ddc/DailyDelhiClimateTest.csv'
df = pd.read_csv(file_path, parse_dates=['date'])

# Set date as index and sort
df.set_index('date', inplace=True)
df.sort_index(inplace=True)

# Plot mean temperature over time
plt.figure(figsize=(10, 5))
plt.plot(df['meantemp'], label='Mean Temperature')
plt.title('Daily Mean Temperature in Delhi')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()

# Train-test split
train = df['meantemp'][:-10]
test = df['meantemp'][-10:]

# Fit ARIMA model (you can tune order=(p,d,q))
model = ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()

# Forecast the next 10 days
forecast = model_fit.forecast(steps=10)

# Plot actual vs predicted
plt.figure(figsize=(10, 5))
plt.plot(test.index, test.values, label='Actual')
plt.plot(test.index, forecast, label='Forecast', linestyle='--', color='red')
plt.title('Forecast vs Actual Temperature')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()

# Print RMSE
rmse = np.sqrt(mean_squared_error(test, forecast))
print("Root Mean Squared Error (RMSE):", rmse)
