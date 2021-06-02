import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from RawDataManagement import df1
from RawDataManagement import raw_data


# denoting the percentage of data used to train the model, in this case 71.5
split_fraction = 0.715
# gives the number of rows that will be used for training in this case 2067
train_split = int(split_fraction * int(df1.shape[0]))


# setting values to be used in the process
past = 720
future = 72
learning_rate = 0.001
batch_size = 256
epochs = 10
step= 1

# features selected in feature selection file, has to be rewritten to use that output as input, headers arent maintained
# in features dataframe
print( "The selected parameters are: TempMinAbs_1, TempProm_1, TempMinAbs_2, TempMinAbs_3, TempProm_3, TempMinAbs_4, "
       "TempProm_4, TempMinAbs_5, TempMinAbs_7, TempProm_7, TempMinAbs_8, TempProm_8, TempMinAbs_9, TempMinAbs_10, "
       "TempProm_10  ")

features = df1[['TempMinAbs_1', 'TempProm_1', 'TempMinAbs_2', 'TempMinAbs_3', 'TempProm_3', 'TempMinAbs_4',
                'TempProm_4', 'TempMinAbs_5', 'TempMinAbs_7', 'TempProm_7', 'TempMinAbs_8', 'TempProm_8',
                'TempMinAbs_9','TempMinAbs_10', 'TempProm_10']]

#features.index = raw_data['Día'] this doesnt work because the train val split uses the indexes,
# how did this work in example?
features.columns = ['TempMinAbs_1', 'TempProm_1', 'TempMinAbs_2', 'TempMinAbs_3', 'TempProm_3', 'TempMinAbs_4',
                    'TempProm_4', 'TempMinAbs_5', 'TempMinAbs_7', 'TempProm_7', 'TempMinAbs_8', 'TempProm_8',
                    'TempMinAbs_9','TempMinAbs_10', 'TempProm_10']

# creates a data frame of train data with the number of rows that will be used for training, in this case 2067 but the
# indexing is 0 to 2066. Also creates a data frame with validation data starting at index 2067 until the end
train_data = features.loc[0: train_split - 1]
val_data = features.loc[train_split:]

start = past + future
end = start + train_split

x_train = train_data.drop(train_data.columns[0], axis=1)
x_train = x_train.to_numpy()
y_train = train_data[1].to_numpy()

sequence_length = int(past / step)

dataset_train = keras.preprocessing.timeseries_dataset_from_array(
    x_train,
    y_train,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)

x_end = len(val_data) - past - future

label_start = train_split + past + future

x_val = val_data.drop(train_data.columns[0], axis=1)
x_val = x_val.to_numpy()
y_val = val_data[1].to_numpy()

dataset_val = keras.preprocessing.timeseries_dataset_from_array(
    x_val,
    y_val,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)


for batch in dataset_train.take(1):
    inputs, targets = batch

print("Input shape:", inputs.numpy().shape)
print("Target shape:", targets.numpy().shape)

inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
lstm_out = keras.layers.LSTM(32)(inputs)
outputs = keras.layers.Dense(1)(lstm_out)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
model.summary()



