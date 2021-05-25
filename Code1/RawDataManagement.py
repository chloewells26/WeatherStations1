import pandas as pd
import numpy as np
import os
from sklearn import preprocessing

# establishing directory
BaseDir = os.path.split(os.path.split(__file__)[0])[0]

# load data, converts csv into pandas dataframe
raw_data = pd.read_csv(BaseDir + '/Data/CompleteData1.csv')


# create a function to convert wind directions to degrees
def dir_to_deg(direction):
    if direction == 'N':
        return 0
    if direction == 'NNE':
        return 22.5
    if direction == 'NE':
        return 45
    if direction == 'ENE':
        return 67.5
    if direction == 'E':
        return 90
    if direction == 'ESE':
        return 112.5
    if direction == 'SE':
        return 135
    if direction == 'SSE':
        return 157.5
    if direction == 'S':
        return 180
    if direction == 'SSW':
        return 202.5
    if direction == 'SW':
        return 225
    if direction == 'WSW':
        return 247.5
    if direction == 'W':
        return 270
    if direction == 'WNW':
        return 292.5
    if direction == 'NW':
        return 315
    if direction == 'NNW':
        return 337.5


# applying function to convert wind direction to degrees
raw_data['DireccionVientoMax_1'] = raw_data['DireccionVientoMax_1'].apply(dir_to_deg)
raw_data['DireccionVientoMax_2'] = raw_data['DireccionVientoMax_2'].apply(dir_to_deg)
raw_data['DireccionVientoMax_3'] = raw_data['DireccionVientoMax_3'].apply(dir_to_deg)
raw_data['DireccionVientoMax_4'] = raw_data['DireccionVientoMax_4'].apply(dir_to_deg)
raw_data['DireccionVientoMax_5'] = raw_data['DireccionVientoMax_5'].apply(dir_to_deg)
raw_data['DireccionVientoMax_6'] = raw_data['DireccionVientoMax_6'].apply(dir_to_deg)
raw_data['DireccionVientoMax_7'] = raw_data['DireccionVientoMax_7'].apply(dir_to_deg)
raw_data['DireccionVientoMax_8'] = raw_data['DireccionVientoMax_8'].apply(dir_to_deg)
raw_data['DireccionVientoMax_9'] = raw_data['DireccionVientoMax_9'].apply(dir_to_deg)
raw_data['DireccionVientoMax_10'] = raw_data['DireccionVientoMax_10'].apply(dir_to_deg)
raw_data['DireccionVientoMax_11'] = raw_data['DireccionVientoMax_11'].apply(dir_to_deg)

# creating new columns with x and y  wind vectors by combining wind direction and speed
raw_data['VientoX_1'] = (raw_data['VelocidadVientoMax_1']) * np.sin(np.deg2rad(raw_data['DireccionVientoMax_1']))
raw_data['VientoY_1'] = (raw_data['VelocidadVientoMax_1']) * np.cos(np.deg2rad(raw_data['DireccionVientoMax_1']))

raw_data['VientoX_2'] = (raw_data['VelocidadVientoMax_2']) * np.sin(np.deg2rad(raw_data['DireccionVientoMax_2']))
raw_data['VientoY_2'] = (raw_data['VelocidadVientoMax_2']) * np.cos(np.deg2rad(raw_data['DireccionVientoMax_2']))

raw_data['VientoX_3'] = (raw_data['VelocidadVientoMax_3']) * np.sin(np.deg2rad(raw_data['DireccionVientoMax_3']))
raw_data['VientoY_3'] = (raw_data['VelocidadVientoMax_3']) * np.cos(np.deg2rad(raw_data['DireccionVientoMax_3']))

raw_data['VientoX_4'] = (raw_data['VelocidadVientoMax_4']) * np.sin(np.deg2rad(raw_data['DireccionVientoMax_4']))
raw_data['VientoY_4'] = (raw_data['VelocidadVientoMax_4']) * np.cos(np.deg2rad(raw_data['DireccionVientoMax_4']))

raw_data['VientoX_5'] = (raw_data['VelocidadVientoMax_5']) * np.sin(np.deg2rad(raw_data['DireccionVientoMax_5']))
raw_data['VientoY_5'] = (raw_data['VelocidadVientoMax_5']) * np.cos(np.deg2rad(raw_data['DireccionVientoMax_5']))

raw_data['VientoX_6'] = (raw_data['VelocidadVientoMax_6']) * np.sin(np.deg2rad(raw_data['DireccionVientoMax_6']))
raw_data['VientoY_6'] = (raw_data['VelocidadVientoMax_6']) * np.cos(np.deg2rad(raw_data['DireccionVientoMax_6']))

raw_data['VientoX_7'] = (raw_data['VelocidadVientoMax_7']) * np.sin(np.deg2rad(raw_data['DireccionVientoMax_7']))
raw_data['VientoY_7'] = (raw_data['VelocidadVientoMax_7']) * np.cos(np.deg2rad(raw_data['DireccionVientoMax_7']))

raw_data['VientoX_8'] = (raw_data['VelocidadVientoMax_8']) * np.sin(np.deg2rad(raw_data['DireccionVientoMax_8']))
raw_data['VientoY_8'] = (raw_data['VelocidadVientoMax_8']) * np.cos(np.deg2rad(raw_data['DireccionVientoMax_8']))

raw_data['VientoX_9'] = (raw_data['VelocidadVientoMax_9']) * np.sin(np.deg2rad(raw_data['DireccionVientoMax_9']))
raw_data['VientoY_9'] = (raw_data['VelocidadVientoMax_9']) * np.cos(np.deg2rad(raw_data['DireccionVientoMax_9']))

raw_data['VientoX_10'] = (raw_data['VelocidadVientoMax_10']) * np.sin(np.deg2rad(raw_data['DireccionVientoMax_10']))
raw_data['VientoY_10'] = (raw_data['VelocidadVientoMax_10']) * np.cos(np.deg2rad(raw_data['DireccionVientoMax_10']))

raw_data['VientoX_11'] = (raw_data['VelocidadVientoMax_11']) * np.sin(np.deg2rad(raw_data['DireccionVientoMax_11']))
raw_data['VientoY_11'] = (raw_data['VelocidadVientoMax_11']) * np.cos(np.deg2rad(raw_data['DireccionVientoMax_11']))

# drop original wind speed and direction values
raw_data = raw_data.drop(['VelocidadVientoMax_1', 'DireccionVientoMax_1', 'VelocidadVientoMax_2', 'DireccionVientoMax_2'
                             , 'VelocidadVientoMax_3', 'DireccionVientoMax_3', 'VelocidadVientoMax_4',
                          'DireccionVientoMax_4', 'VelocidadVientoMax_5', 'DireccionVientoMax_5', 'VelocidadVientoMax_6'
                             , 'DireccionVientoMax_6', 'VelocidadVientoMax_7', 'DireccionVientoMax_7',
                          'VelocidadVientoMax_8', 'DireccionVientoMax_8', 'VelocidadVientoMax_9', 'DireccionVientoMax_9'
                             , 'VelocidadVientoMax_10', 'DireccionVientoMax_10', 'VelocidadVientoMax_11',
                          'DireccionVientoMax_11'], axis=1)

# create a data frame with columns in the preferred order
x = raw_data[['TempMinAbs_1', 'TempProm_1', 'TempMaxAbs_1', 'Hum_1', 'Precipitacion_1', 'RadSolar_1',
                 'RadSolarMaxAbs_1', 'IndiceUV_1', 'IndiceUVMaxAbs_1', 'VientoX_1', 'VientoY_1',
                 'TempMinAbs_2', 'TempProm_2', 'TempMaxAbs_2', 'Hum_2', 'Precipitacion_2', 'RadSolar_2',
                 'RadSolarMaxAbs_2', 'IndiceUV_2', 'IndiceUVMaxAbs_2', 'VientoX_2', 'VientoY_2', 'TempMinAbs_3',
                 'TempProm_3', 'TempMaxAbs_3', 'Hum_3', 'Precipitacion_3', 'RadSolar_3', 'RadSolarMaxAbs_3',
                 'IndiceUV_3', 'IndiceUVMaxAbs_3', 'VientoX_3', 'VientoY_3', 'TempMinAbs_4', 'TempProm_4',
                 'TempMaxAbs_4', 'Hum_4', 'Precipitacion_4', 'RadSolar_4', 'RadSolarMaxAbs_4', 'IndiceUV_4',
                 'IndiceUVMaxAbs_4', 'VientoX_4', 'VientoY_4', 'TempMinAbs_5', 'TempProm_5', 'TempMaxAbs_5',
                 'Hum_5', 'Precipitacion_5', 'RadSolar_5', 'RadSolarMaxAbs_5', 'IndiceUV_5', 'IndiceUVMaxAbs_5',
                 'VientoX_5', 'VientoY_5', 'TempMinAbs_6', 'TempProm_6', 'TempMaxAbs_6', 'Hum_6',
                 'Precipitacion_6', 'RadSolar_6', 'RadSolarMaxAbs_6', 'IndiceUV_6', 'IndiceUVMaxAbs_6', 'VientoX_6',
                 'VientoY_6', 'TempMinAbs_7', 'TempProm_7', 'TempMaxAbs_7', 'Hum_7', 'Precipitacion_7',
                 'RadSolar_7', 'RadSolarMaxAbs_7', 'IndiceUV_7', 'IndiceUVMaxAbs_7', 'VientoX_7', 'VientoY_7',
                 'TempMinAbs_8', 'TempProm_8', 'TempMaxAbs_8', 'Hum_8', 'Precipitacion_8', 'RadSolar_8',
                 'RadSolarMaxAbs_8', 'IndiceUV_8', 'IndiceUVMaxAbs_8', 'VientoX_8', 'VientoY_8', 'TempMinAbs_9',
                 'TempProm_9', 'TempMaxAbs_9', 'Hum_9', 'Precipitacion_9', 'RadSolar_9', 'RadSolarMaxAbs_9',
                 'IndiceUV_9', 'IndiceUVMaxAbs_9', 'VientoX_9', 'VientoY_9', 'TempMinAbs_10', 'TempProm_10',
                 'TempMaxAbs_10', 'Hum_10', 'Precipitacion_10', 'RadSolar_10', 'RadSolarMaxAbs_10', 'IndiceUV_10',
                 'IndiceUVMaxAbs_10', 'VientoX_10', 'VientoY_10', 'TempMinAbs_11', 'TempProm_11', 'TempMaxAbs_11',
                 'Hum_11', 'Precipitacion_11', 'RadSolar_11', 'RadSolarMaxAbs_11', 'IndiceUV_11', 'IndiceUVMaxAbs_11',
                 'VientoX_11', 'VientoY_11']]


# normalizing data by sample and reconverting into dataframe
X_train = x.to_numpy()
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)



df1 = pd.DataFrame(X_train_minmax)
df1.columns = ['TempMinAbs_1', 'TempProm_1', 'TempMaxAbs_1', 'Hum_1', 'Precipitacion_1', 'RadSolar_1',
                 'RadSolarMaxAbs_1', 'IndiceUV_1', 'IndiceUVMaxAbs_1', 'VientoX_1', 'VientoY_1',
                 'TempMinAbs_2', 'TempProm_2', 'TempMaxAbs_2', 'Hum_2', 'Precipitacion_2', 'RadSolar_2',
                 'RadSolarMaxAbs_2', 'IndiceUV_2', 'IndiceUVMaxAbs_2', 'VientoX_2', 'VientoY_2', 'TempMinAbs_3',
                 'TempProm_3', 'TempMaxAbs_3', 'Hum_3', 'Precipitacion_3', 'RadSolar_3', 'RadSolarMaxAbs_3',
                 'IndiceUV_3', 'IndiceUVMaxAbs_3', 'VientoX_3', 'VientoY_3', 'TempMinAbs_4', 'TempProm_4',
                 'TempMaxAbs_4', 'Hum_4', 'Precipitacion_4', 'RadSolar_4', 'RadSolarMaxAbs_4', 'IndiceUV_4',
                 'IndiceUVMaxAbs_4', 'VientoX_4', 'VientoY_4', 'TempMinAbs_5', 'TempProm_5', 'TempMaxAbs_5',
                 'Hum_5', 'Precipitacion_5', 'RadSolar_5', 'RadSolarMaxAbs_5', 'IndiceUV_5', 'IndiceUVMaxAbs_5',
                 'VientoX_5', 'VientoY_5', 'TempMinAbs_6', 'TempProm_6', 'TempMaxAbs_6', 'Hum_6',
                 'Precipitacion_6', 'RadSolar_6', 'RadSolarMaxAbs_6', 'IndiceUV_6', 'IndiceUVMaxAbs_6', 'VientoX_6',
                 'VientoY_6', 'TempMinAbs_7', 'TempProm_7', 'TempMaxAbs_7', 'Hum_7', 'Precipitacion_7',
                 'RadSolar_7', 'RadSolarMaxAbs_7', 'IndiceUV_7', 'IndiceUVMaxAbs_7', 'VientoX_7', 'VientoY_7',
                 'TempMinAbs_8', 'TempProm_8', 'TempMaxAbs_8', 'Hum_8', 'Precipitacion_8', 'RadSolar_8',
                 'RadSolarMaxAbs_8', 'IndiceUV_8', 'IndiceUVMaxAbs_8', 'VientoX_8', 'VientoY_8', 'TempMinAbs_9',
                 'TempProm_9', 'TempMaxAbs_9', 'Hum_9', 'Precipitacion_9', 'RadSolar_9', 'RadSolarMaxAbs_9',
                 'IndiceUV_9', 'IndiceUVMaxAbs_9', 'VientoX_9', 'VientoY_9', 'TempMinAbs_10', 'TempProm_10',
                 'TempMaxAbs_10', 'Hum_10', 'Precipitacion_10', 'RadSolar_10', 'RadSolarMaxAbs_10', 'IndiceUV_10',
                 'IndiceUVMaxAbs_10', 'VientoX_10', 'VientoY_10', 'TempMinAbs_11', 'TempProm_11', 'TempMaxAbs_11',
                 'Hum_11', 'Precipitacion_11', 'RadSolar_11', 'RadSolarMaxAbs_11', 'IndiceUV_11', 'IndiceUVMaxAbs_11',
                 'VientoX_11', 'VientoY_11']


print(df1)
