import pandas as pd
import tensorflow
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from RawDataManagement import df1
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense


# features selected in feature selection file, has to be rewritten to use that output as input, headers arent maintained
# in features dataframe
print("The selected parameters are: TempMinAbs_1, TempProm_1, TempMinAbs_2, TempMinAbs_3, TempProm_3, TempMinAbs_4, "
      "TempProm_4, TempMinAbs_5, TempMinAbs_7, TempProm_7, TempMinAbs_8, TempProm_8, TempMinAbs_9, TempMinAbs_10, "
      "TempProm_10")

target = df1[['TempMinAbs_1']]
target.columns = ['TempMinAbs_1']

features = df1[['TempMinAbs_1', 'TempProm_1', 'TempMinAbs_2', 'TempMinAbs_3', 'TempProm_3', 'TempMinAbs_4',
                'TempProm_4', 'TempMinAbs_5', 'TempMinAbs_7', 'TempProm_7', 'TempMinAbs_8', 'TempProm_8',
                'TempMinAbs_9', 'TempMinAbs_10', 'TempProm_10']]

# features.index = raw_data['Día'] this doesnt work because the train val split uses the indexes,
# how did this work in example?
features.columns = ['TempMinAbs_1', 'TempProm_1', 'TempMinAbs_2', 'TempMinAbs_3', 'TempProm_3', 'TempMinAbs_4',
                    'TempProm_4', 'TempMinAbs_5', 'TempMinAbs_7', 'TempProm_7', 'TempMinAbs_8', 'TempProm_8',
                    'TempMinAbs_9','TempMinAbs_10', 'TempProm_10']
features = features.fillna(-1)


n_features = len(features.columns)
M = 212
N = 16
M_N = M + N
batch_size = 1
# 228

# training data is M + N length with no data gaps in the target, in this case 228 rows
X_Train = features[1324:1535]
X_Train = X_Train.to_numpy()
X_Train = X_Train.reshape(batch_size, X_Train.shape[0], X_Train.shape[1])

y_Train = target[1536:1552]
y_Train = y_Train.to_numpy()
y_Train = y_Train.reshape(batch_size, y_Train.shape[0], y_Train.shape[1])

# 212 rows of data with no gaps in corresponding target that precede a data gap of 16 in target
X_in = features[1096:1307]
X_in = X_in.to_numpy()
X_in = X_in.reshape(batch_size, X_in.shape[0], X_in.shape[1])

n_steps_in, n_steps_out = 211, 16

n_features = 15

# print(X_Train.shape[0])   1
# print(X_Train.shape[1])  228
# print(X_in.shape[0])      1
# print(X_in.shape[1])     211


model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X_Train, y_Train, epochs=30, verbose=2)
# demonstrate prediction
out = model.predict(X_in, verbose=2)
print(out)


observed = X_in[:, :, 0]
predicted = out.reshape(out.shape[1])
observed = observed.reshape(observed.shape[1])
print(observed.shape)
print(observed)

total = np.append(observed, predicted)
plt.plot(total)
# label x axis
plt.xlabel('timesteps')
# label y axis
plt.ylabel('scaled values')
# Set a title of the current axes.
plt.title('Temp Min Abs Barreneche')
# show a legend on the plot
# plt.legend()
# Display a figure.
plt.show()

