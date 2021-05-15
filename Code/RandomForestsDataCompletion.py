import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os

BaseDir = os.path.split(os.path.split(__file__)[0])[0]
# load data, converts csv into pandas dataframe
cd = pd.read_csv(BaseDir + '/Data/CompleteData.csv')


# create a function to convert date string to float, with day 1 being the first day of 2011
def string_to_float(date_string):
    days = int(date_string[-2:])
    months = int(date_string[-5:-3]) * 30
    years = (int(date_string[:4]) - 2011) * 365
    total_days = days + months + years
    return total_days


# apply the function to the date column
cd['Dia'] = cd['Dia'].apply(string_to_float)


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


# apply the function to the wind direction column
cd['DireccionVientoMax'] = cd['DireccionVientoMax'].apply(dir_to_deg)

# create new columns for x and y values of wind speed and direction vector
cd['VientoX'] = (cd['VelocidadVientoMax']) * np.sin(np.deg2rad(cd['DireccionVientoMax']))
cd['VientoY'] = (cd['VelocidadVientoMax']) * np.cos(np.deg2rad(cd['DireccionVientoMax']))

# drop original wind speed and direction values
cd = cd.drop(['VelocidadVientoMax', 'DireccionVientoMax'], axis=1)

# handle missing values
cd = cd.dropna()

# create a list to store generated values
output = []

# create values for x, leaving out the date column and the location column
column_names = ['TempMinAbs', 'TempProm', 'TempMaxAbs', 'Hum', 'Precipitacion', 'RadSolar', 'RadSolarMaxAbs', 'IndiceUV',
                'IndiceUVMaxAbs', 'VientoX', 'VientoY']

# start a for loop  containing the random forests
for x in column_names:
    X_train, X_test, Y_train, Y_test = train_test_split(cd.drop(labels=[x], axis =1), cd[x].values,
                                                        test_size=0.4)
    rf = RandomForestRegressor(n_estimators=10, random_state=30)
    rf.fit(X_train, Y_train)
    predictions = rf.predict(X_test)
    output.append(predictions)               # append the output of the random forest to the list
    errors = abs(predictions - Y_test)
    print('Mean Absolute Error:', np.round(np.mean(errors), 2), 'degrees.')



PredictionData = pd.DataFrame(output, index=['TempMinAbs_Pred', 'TempProm_Pred', 'TempMaxAbs_Pred', 'Hum_Pred',
                              'Precipitacion_Pred', 'RadSolar_Pred', 'RadSolarMaxAbs_Pred', 'IndiceUV_Pred',
                              'IndiceUVMaxAbs_Pred', 'VientoX_Pred', 'VientoY_Pred'])

PredictionData = PredictionData.T

ActualData = cd.drop(['Dia', 'location'], axis=1)






#pd.set_option('display.max_columns', None)
#print(cd.head())
