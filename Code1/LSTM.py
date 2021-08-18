import pandas as pd
import tensorflow
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from RawDataManagement import df1
from RawDataManagement import df2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Separate dates for future plotting
train_dates = pd.to_datetime(df2['Día'])

# Variables for training
cols = list(df2[['TempMinAbs_1', 'TempProm_1', 'TempMinAbs_2', 'TempMinAbs_3', 'TempProm_3', 'TempMinAbs_4',
                'TempProm_4', 'TempMinAbs_5', 'TempMinAbs_7', 'TempProm_7', 'TempMinAbs_8', 'TempProm_8',
                'TempMinAbs_9', 'TempMinAbs_10', 'TempProm_10']])

df_for_training = df2[cols].astype(float)

# LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
# normalize the dataset *standard scaler converts to array for some reason? dont want to fill NaN values before scaling
scaler = StandardScaler()
scaler = scaler.fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)

df3 = pd.DataFrame(df_for_training_scaled)
df3 = df3.fillna(2)
df_for_training_scaled = df3.to_numpy()


# As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features.
# In this example, the n_features is 2. We will make timesteps = 3.
# With this, the resultant n_samples is 5 (as the input data has 9 rows).
trainX = []
trainY = []

n_future = 1  # Number of days we want to predict into the future
n_past = 14  # Number of past days we want to use to predict the future

for i in range(n_past, len(df_for_training_scaled) - n_future + 1):
    trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

trainX, trainY = np.array(trainX), np.array(trainY)

print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))

# define Autoencoder model

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(trainY.shape[1]))

model.compile(optimizer='adam', loss='mse')
model.summary()

# fit model
history = model.fit(trainX, trainY, epochs=25, batch_size=16, validation_split=0.2, verbose=1)

# plt.plot(history.history['loss'], label='Training loss')
# plt.plot(history.history['val_loss'], label='Validation loss')
# plt.legend()

# Forecasting...
# Start with the last day in training date and predict future...
n_future = 90  # Redefining n_future to extend prediction dates beyond original n_future dates...
forecast_period_dates = pd.date_range(list(train_dates)[-1], periods=n_future, freq='1d').tolist()

forecast = model.predict(trainX[-n_future:])  # forecast

# Perform inverse transformation to rescale back to original range
# Since we used 5 variables for transform, the inverse expects same dimensions
# Therefore, let us copy our values 5 times and discard them after inverse transform
forecast_copies = np.repeat(forecast, df_for_training.shape[1], axis=-1)
y_pred_future = scaler.inverse_transform(forecast_copies)[:, 0]

# Convert timestamp to date
forecast_dates = []
for time_i in forecast_period_dates:
    forecast_dates.append(time_i.date())

df_forecast = pd.DataFrame({'Día': np.array(forecast_dates), 'TempMinAbs_1': y_pred_future})
df_forecast['Día'] = pd.to_datetime(df_forecast['Día'])

original = df2[['Día', 'TempMinAbs_1']]
original['Día'] = pd.to_datetime(original['Día'])
original = original.loc[original['Día'] >= '10/5/2016']

sns.lineplot(original['Día'], original['TempMinAbs_1'])
sns.lineplot(df_forecast['Día'], df_forecast['TempMinAbs_1'])
plt.show()


loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



