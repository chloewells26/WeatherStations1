from sklearn import preprocessing
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

#load the data
X, y = load_breast_cancer(return_X_y=True)
data = load_breast_cancer()
print(type(data))
print(type(X))
print(type(y))

# define random forest classifier
'''
n_jobs is number of jobs to run in parallel and default is one. Different jobs can be  like "fit(X, y, sample_weight = " 
which builds a forest of trees from the data set, 'predict(x)' which predicts the class of x, 'decision_path(x)' which
will return the decision path that led to the classification of x or 'apply(x)' which will return the index of the leaf
that x ends up in
X is array like but in this case samples with features and y is target which is class (benign or malignant)
this dataset it a binary classifier data set ie samples can be classified one of two ways (in this case malignant or 
benign) class weight = balanced means that if there are 10x as many samples in class 0 compared to class 1, class one 
data will be replicated 10 times to balance the data this will capture more class events (higher class 1 recall) but 
also you are more likely to get false alerts (lower class 1 precision)
max_depth = 5 clarifies the number of nodes in the tree, a node is anywhere where the path splits, you can have one 
qualifying feature in a node (ex is it red) or many qualifying features at one node (ex is it red, round and heavy)
in random forest, trees are randomly assigned features to use as qualifiers at their nodes, trees are trained on 
different sets of data due to bagging but also learn to make decisions based on different features which leads to 
uncorrelated trees that buffer and protect one false decision from proliferating in the forest'pruning a tree' or 
specifying max depth helps to avoid over fitting a model. Over fitting means to make an overly complex model to explain 
the idiosyncrasies (peculiarities) of the particular data set that don't represent the general situation being modelled
'''
forest = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
forest.fit(X, y)


# define Boruta feature selection method
'''
boruta can be used with various methods (XGboost, random forest, regression) so first argument is to clarify we are
using with random forest, number of estimators set to auto means number of trees used in decision process will
be chosen automatically based on data size, more trees gives higher accuracy but slower code
the verbosity of the model is how much you want to be shown during the training process (verbose = 0 shows you nothing)
because we want to see each iteration we put it at verbose = 2 which will give iteration number, number of confirmed
and number of rejected for each iteration.
not sure how to change number of iterations and reject/confirm conditions
setting random state to an integer ensures that the same sequence of random numbers are generated when you run the code,
it doesnt change how random it is but assigns a state to that combination of random to make results replicable
'''
feat_selector = BorutaPy(forest, n_estimators='auto', verbose=2, random_state=1)

# find all relevant features
'''
same as when we use the random forest classifier, we are building our forest for feature selection now, not totally sure
if you would need ot have your final model first? not even sure why its here now
'''
feat_selector.fit(X, y)

# check selected features: .support_ attribute is a boolean array that answers — should feature should be kept?
feat_selector.support_

# check ranking of features: .ranking_ attribute is an int array for the rank (1 is best feature(s))
feat_selector.ranking_

# call transform() on X to filter it down to selected features
'''
transform(X) method applies the suggestions and returns an array of adjusted data. You could just let Boruta manage the 
entire ordeal. So this would give you a new data set called X_filtered that contained only the selected features
'''
X_filtered = feat_selector.transform(X)

'''
this next section is just to look at the results of borutas process
the zip function combines multiple iterables into a 'zip object' which can then be read as a list
an iterable is an object that contains a countable number of features, in this case the iterators are 
data.feature_names (what library is feature_names from?), feat_selector.ranking_ and feat_selector.support_
you can iterate through, lists, tuples, dictionaries, sets and even strings; if you iterate through a string you get
the letters in the string
the zip makes the three iterables into a single iterable and the list makes it into a list, if the dimensions of the the
iterables being combined is mismatched then the new iterable will have the length of the shortest input iterable
'''
# zip my names, ranks, and decisions in a single iterable
feature_ranks = list(zip(data.feature_names,
                         feat_selector.ranking_,
                         feat_selector.support_))

'''
feature_ranks is a list, the for loop is to assign the explanations 'feature, rank and keep' to each line of  the list
feat[0] refers to data.feature_names, feat[1] refers to feat_selector.ranking_ and feat[2] refers to 
feat_selector_support. 
the feature name is obviously feature name, rank of 1 is beast feature, keep is either true or false
not sure why ranking is either 1,2 or 3 here? anything other than 1 is not kept -> 1 is keep, 2 is unsure, 3 is reject
'''
# iterate through and print out the results
for feat in feature_ranks:
    print('Feature: {:<25} Rank: {},  Keep: {}'.format(feat[0], feat[1], feat[2]))





