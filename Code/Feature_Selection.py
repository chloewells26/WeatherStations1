from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import Bunch
from boruta import BorutaPy as bp
import os
import pandas as pd
import numpy as np
import timeit

start = timeit.default_timer()

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
cd = cd.to_numpy()

X = np.delete(cd, 2, 1)
y = cd[:, 2]

cd_bunch = Bunch(data=X, target=y, feature_names=np.array(['Dia', 'TempProm', 'TempMaxAbs', 'Hum', 'Precipitacion', 'RadSolar', 'RadSolarMaxAbs', 'IndiceUV',
                'IndiceUVMaxAbs', 'VientoX', 'VientoY']), DESCR='hi guys whats up')


rf_model = RandomForestRegressor(n_jobs=4, oob_score=True)
feat_selector = bp(rf_model, n_estimators='auto', verbose=2, max_iter=50)
feat_selector.fit(X, y.ravel())


# check selected features: .support_ attribute is a boolean array that answers — should feature should be kept?
feat_selector.support_
# check ranking of features: .ranking_ attribute is an int array for the rank (1 is best feature(s))
feat_selector.ranking_
X_filtered = feat_selector.transform(X)
feature_ranks = list(zip(cd_bunch.feature_names,
                         feat_selector.ranking_,
                         feat_selector.support_))
for feat in feature_ranks:
    print('Feature: {:<25} Rank: {},  Keep: {}'.format(feat[0], feat[1], feat[2]))


stop = timeit.default_timer()

print('Time: ', stop - start)

#this is a test