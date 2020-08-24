

# %%

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer

# %%

class ItemsSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        if isinstance(features, list):
            self.features = features
        else:
            self.features = [features]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        return X.loc[:,self.features]


class CategoricalImputer(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        if isinstance(features, list):
            self.features = features
        else:
            self.features = [features]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.features:
            X[feature] = X[feature].fillna('Missing')
        return X


class NumericalImputer(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        if isinstance(features, list):
            self.features = features
        else:
            self.features = [features]

    def fit(self, X, y=None):
        self.imputer_dict = {}
        for feature in self.features:
            self.imputer_dict[feature] = X[feature].median()
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.features:
            X[feature] = X[feature].fillna(self.imputer_dict[feature])
        return X



class TemporalVariableTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features, reference_feature):
        self.reference_feature = reference_feature

        if isinstance(features, list):
            self.features = [feature for feature in features if feature != self.reference_feature]
        else:
            self.features = [features]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.features:
            X[feature] = X[self.reference_feature] - X[feature]
        X = X.drop(self.reference_feature, axis=1)
        return X


class OtherLabelCategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, features, rare_perc=0.01):
        self.rare_perc = rare_perc
        if isinstance(features, list):
            self.features = features
        else:
            self.features = [features]

    def fit(self, X, y=None):
        self.frequent_labels_dict = {}
        for feature in self.features:
            t = X[feature].value_counts() / X.shape[0]
            self.frequent_labels_dict[feature] = t[t >= self.rare_perc].index
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.features:
            X[feature] = np.where(X[feature].isin(self.frequent_labels_dict[feature]), X[feature], 'Other')
        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        if isinstance(features, list):
            self.features = features
        else:
            self.features = [features]

    def fit(self, X, y):
        self.ordered_labels_dict = {}
        tmp = pd.concat([X, y], axis=1)
        for feature in self.features:
            ordered_labels = tmp.groupby(feature)['SalePrice'].median().sort_values().index
            ordered_labels = {k: i for i, k in enumerate(ordered_labels, 0)}
            self.ordered_labels_dict[feature] = ordered_labels
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.features:
            X[feature] = X[feature].map(self.ordered_labels_dict[feature])
        return X

class SelectedFeaturesPowerTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        self.transformers_dict = {}
        for feature in self.features:
            pt = PowerTransformer()
            pt.fit(np.array(X[feature]).reshape(-1, 1))
            self.transformers_dict[feature] = pt
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.features:
            X[feature] = self.transformers_dict[feature].transform(np.array(X[feature]).reshape(-1, 1))
        return X
