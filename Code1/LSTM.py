import pandas as pd
import tensorflow
from tensorflow import keras
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

# training data is M + N length with no data gaps in the target, in this case 228 rows
X_Train = features[1324:1552]
X_Train = X_Train.to_numpy()
X_Train = X_Train.reshape(1, 228, 15)

y_Train = target[1324:1552]
y_Train = y_Train.to_numpy()
y_Train = y_Train.reshape(1, 228, 1)
# 212 rows of data with no gaps in corresponding target that precede a data gap of 16 in target
X_in = features[1096:1307]
X_in = X_in.to_numpy()
X_in = X_in.reshape(1, 211, 15)

n_steps_in, n_steps_out = 228, 16

n_features = 15

model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X_Train, y_Train, epochs=30, verbose=2)
# demonstrate prediction
out = model.predict(X_in, verbose=2)
print(out)



