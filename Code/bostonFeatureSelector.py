from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy as bp
from sklearn.datasets import load_boston
import numpy as np
boston = load_boston()
X = boston.data
y = boston.target

print(type(boston))
print(type(boston.data))
print(type(boston.feature_names))
print(type(boston.DESCR))

rf_model = RandomForestRegressor(n_jobs= 4,oob_score= True)
feat_selector = bp(rf_model,n_estimators = 'auto', verbose= 0,max_iter= 4)
feat_selector.fit(X, y)
selected_features = [boston.feature_names[i] for i, x in enumerate(feat_selector.support_) if x]
print(selected_features)
