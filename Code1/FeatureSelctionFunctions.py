import RawDataManagement as rdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import Bunch
from boruta import BorutaPy as bp
import numpy as np



def pearson_selection(feature_name):
    corr = rdm.df1.corr().loc[feature_name]
    named = corr.rename('Correlations of ' + feature_name)
    signif = named[abs(named) >= 0.5]
    print(signif)


pearson_selection('TempMinAbs_1')

'''
def boruta_selection(feature_name):
    df2 = rdm.df1
    X = df2.drop([feature_name], axis=1)
    y = df2[feature_name]
    X = X.to_numpy()
    y = y.to_numpy()
    cd_bunch = Bunch(data=X, target=y, feature_names=np.array(rdm.x.columns))
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


boruta_selection('TempMinAbs_1')



def show_heatmap(data):
    plt.matshow(rdm.df1.corr())
    plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(range(data.shape[1]), data.columns, fontsize=14)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title("Feature Correlation Heatmap", fontsize=14)
    plt.show()


show_heatmap(rdm.df1)
'''