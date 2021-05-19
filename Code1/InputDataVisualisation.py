import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import RawDataManagement as rdm

# define the lines
x1 = rdm.data['Día']
y1 = rdm.data['TempMinAbs_1']
y2 = rdm.data['TempMinAbs_2']
y3 = rdm.data['TempMinAbs_3']
y4 = rdm.data['TempMinAbs_4']
y5 = rdm.data['TempMinAbs_5']
y6 = rdm.data['TempMinAbs_6']
y7 = rdm.data['TempMinAbs_7']
y8 = rdm.data['TempMinAbs_8']
y9 = rdm.data['TempMinAbs_9']
y10 = rdm.data['TempMinAbs_10']
y11 = rdm.data['TempMinAbs_11']

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
plt.xticks(np.arange(0, len(x1)+1, 110))
# label y axis
plt.ylabel('Temp Min Abs (\u00b0C)')
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