import matplotlib.pyplot as plt
import numpy as np
import RawDataManagement as rdm

# define the lines
x1 = rdm.raw_data['Día']
y1 = rdm.df1['TempMinAbs_1']
y2 = rdm.df1['TempProm_1']
y3 = rdm.df1['TempMaxAbs_1']
y4 = rdm.df1['Hum_1']
y5 = rdm.df1['Precipitacion_1']
y6 = rdm.df1['RadSolar_1']
y7 = rdm.df1['RadSolarMaxAbs_1']
y8 = rdm.df1['IndiceUV_1']
y9 = rdm.df1['IndiceUVMaxAbs_1']
y10 = rdm.df1['VientoX_1']
y11 = rdm.df1['VientoY_1']

# plotting the lines
plt.plot(x1, y1, label='TempMinAbs_1', alpha=0.5)
plt.plot(x1, y2, label='TempProm_1', alpha=0.5)
plt.plot(x1, y3, label='TempMaxAbs_1', alpha=0.5)
plt.plot(x1, y4, label='Hum_1', alpha=0.5)
plt.plot(x1, y5, label='Precipitacion_1', alpha=0.5)
plt.plot(x1, y6, label='RadSolar_1', alpha=0.5)
plt.plot(x1, y7, label='RadSolarMaxAbs_1', alpha=0.5)
plt.plot(x1, y8, label='IndiceUV_1', alpha=0.5)
plt.plot(x1, y9, label='IndiceUVMaxAbs_1', alpha=0.5)
plt.plot(x1, y10, label='VientoX_1', alpha=0.5)
plt.plot(x1, y11, label='VientoY_1', alpha=0.5)
# label x axis
plt.xlabel('Día')
plt.xticks(rotation=45)
plt.xticks(np.arange(0, len(x1)+1, 110))
# label y axis
plt.ylabel('Barreneche')
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