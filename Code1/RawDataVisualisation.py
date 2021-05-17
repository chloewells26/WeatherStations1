import matplotlib as plt
import pandas as pd
import os
import numpy as np

BaseDir = os.path.split(os.path.split(__file__)[0])[0]
# load data, converts csv into pandas dataframe
cd = pd.read_csv(BaseDir + '/Data/CompleteData1.csv')
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


# apply the function to the wind direction column **need to find a way to apply to all wind direction
# columns easily, right now only applied to first wind direction column in barreneche
cd['DireccionVientoMax_1'] = cd['DireccionVientoMax_1'].apply(dir_to_deg)

# create new columns for x and y values of wind speed and direction vector
cd['VientoX'] = (cd['VelocidadVientoMax_1']) * np.sin(np.deg2rad(cd['DireccionVientoMax_1']))
cd['VientoY'] = (cd['VelocidadVientoMax_1']) * np.cos(np.deg2rad(cd['DireccionVientoMax_1']))

# drop original wind speed and direction values
cd = cd.drop(['VelocidadVientoMax_1', 'DireccionVientoMax_1'], axis=1)

titles = [
    'Barreneche'
    'El Tablon'
    'Finca Santa Victoria'
    'Los Saminez'
    'Pamesebal'
    'Panajachel'
    'San Jose Chacaya'
    'San Juan la Laguna'
    'San Lucas Toliman'
    'Santa Lucia Utatlan'
    'Santigao Atitlan'
]

features = [
    'Temp Min Abs'
    'Temp Prom'
    'Temp Max Abs'
    'Hum'
    'Precipitacion'
    'Rad Solar'
    'Rad Solar Max Abs'
    'Indice UV'
    'Indice UV Max Abs'
    'Viento X'
    'Viento Y'
]

colours = [
    'blue'
    'orange'
    'green'
    'red'
    'purple'
    'brown'
    'pink'
    'gray'
    'olive'
    'cyan'
    'black'
]