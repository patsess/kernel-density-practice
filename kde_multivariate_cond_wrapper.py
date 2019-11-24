
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
import statsmodels.api as sm


class KDEMultiCondWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, dep_type='c', indep_type='c', bw='normal_reference'):
        self.dep_type = dep_type
        self.indep_type = indep_type
        self.bw = bw

    def fit(self, X, y):
        self.model_ = sm.nonparametric.KDEMultivariateConditional(
            endog=y, exog=X, dep_type=self.dep_type,
            indep_type=self.indep_type, bw=self.bw)

        return self

    def predict(self, X, y):
        return self.model_.pdf(endog_predict=y, exog_predict=X)

    def score(self, X, y, sample_weight=None):
        return np.maximum(1e-100, self.predict(X=X, y=y)).sum()
