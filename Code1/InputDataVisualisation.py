import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

# establishing directory
BaseDir = os.path.split(os.path.split(__file__)[0])[0]

# load data, converts csv into pandas dataframe
cd1 = pd.read_csv(BaseDir + '/Data/CompleteData1.csv')


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


cd1['DireccionVientoMax_1'] = cd1['DireccionVientoMax_1'].apply(dir_to_deg)
cd1['VientoX_1'] = (cd1['VelocidadVientoMax_1']) * np.sin(np.deg2rad(cd1['DireccionVientoMax_1']))
cd1['VientoY_1'] = (cd1['VelocidadVientoMax_1']) * np.cos(np.deg2rad(cd1['DireccionVientoMax_1']))

# drop original wind speed and direction values
cd1 = cd1.drop(['VelocidadVientoMax_1', 'DireccionVientoMax_1'], axis=1)

# define the lines
x1 = cd1['Día']
y1 = cd1['TempMinAbs_1']
y2 = cd1['TempMinAbs_2']
y3 = cd1['TempMinAbs_3']
y4 = cd1['TempMinAbs_4']
y5 = cd1['TempMinAbs_5']
y6 = cd1['TempMinAbs_6']
y7 = cd1['TempMinAbs_7']
y8 = cd1['TempMinAbs_8']
y9 = cd1['TempMinAbs_9']
y10 = cd1['TempMinAbs_10']
y11 = cd1['TempMinAbs_11']

# plotting the lines
plt.plot(x1, y1, label='Barreneche', alpha=0.5)
plt.plot(x1, y2, label='El Tablon', alpha=0.5)
plt.plot(x1, y3, label='Finca Santa Victoria', alpha=0.5)
plt.plot(x1, y4, label='Los Saminez', alpha=0.5)
plt.plot(x1, y5, label='Pamesebal', alpha=0.5)
plt.plot(x1, y6, label='Panajachel', alpha=0.5)
plt.plot(x1, y7, label='San Jose Chacaya', alpha=0.5)
plt.plot(x1, y8, label='San Juan la Laguna', alpha=0.5)
plt.plot(x1, y9, label='San Lucas Toliman', alpha=0.5)
plt.plot(x1, y10, label='Santa Lucia Utatlan', alpha=0.5)
plt.plot(x1, y11, label='Santiago Atitlan', alpha=0.5)
# label x axis
plt.xlabel('Día')
plt.xticks(rotation=45)
plt.xticks(np.arange(0, len(x1)+1, 100))
# label y axis
plt.ylabel('Temp Min Abs (^O C')
# Set a title of the current axes.
plt.title('TempMinAbs')
# show a legend on the plot
plt.legend()
# Display a figure.
plt.show()







'''
# create new columns for x and y values of wind speed and direction vector
cd1['VientoX_1'] = (cd1['VelocidadVientoMax_1']) * np.sin(np.deg2rad(cd1['DireccionVientoMax_1']))
cd1['VientoY_1'] = (cd1['VelocidadVientoMax_1']) * np.cos(np.deg2rad(cd1['DireccionVientoMax_1']))

# drop original wind speed and direction values
cd1 = cd1.drop(['VelocidadVientoMax_1', 'DireccionVientoMax_1'], axis=1)

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

pd.set_option('display.max_rows', None)
print(cd1['DireccionVientoMax_1'])
'''