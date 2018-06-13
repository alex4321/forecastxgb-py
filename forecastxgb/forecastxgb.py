from .xgbar import xgbar, forecast, summary
from sklearn.feature_selection.base import SelectorMixin
import pandas as pd
import numpy as np


class ForecastXGB:
    SEAS_DUMMIES ='dummies'
    SEAS_DECOMPOSE = 'decompose'
    SEAS_FOURIER = 'fourier'
    SEAS_NONE = 'none'
    TREND_NONE = 'none'
    TREND_DIFF = 'differencing'
    NROUNDS_METHOD_CV = 'cv'
    NROUNDS_METHOD_V = 'v'
    NROUNDS_METHOD_MANUAL = 'm'

    def __init__(self, maxlag=None, lambda_=1, K=None,
                 seas_method=SEAS_DUMMIES, trend_method=TREND_NONE,
                 nrounds_method=NROUNDS_METHOD_CV, nrounds_max=100,
                 nfold=None,
                 features_eliminator_builder=None,
                 **kwargs):
        """
        Initialize model
        :param maxlag: maximum lag of features to use
        :type maxlag: int
        :param lambda_:
        :param K:
        :param seas_method: method of building season features (only SEAS_DUMMIES implemented now)
        :type seas_method: str
        :param trend_method: method of removing (and restoring) global trend (only TREND_NONE implemented now)
        :type trend_method: str
        :param nrounds_method: method of choosing best tree count (only NROUNDS_METHOD_CV implemented now)
        :type nrounds_method: str
        :param nrounds_max: max numbers of trees (or just tree number if nrounds_method=NROUNDS_METHOD_MANUAL)
        :type nrounds_max: int
        :param nfold: number of cross-validation rounds (by default will use 10-rounds)
        :type nfold: int
        :param features_eliminator_builder: function which'll build feature selector. \
            Must consume num_boost_round - max tree count and other XGBRegressor params as **kwargs
        :type features_eliminator_builder: (int,)->SelectorMixin
        :param kwargs: custom XGBoost params (e.g. custom objective and other like it)
        """
        self.maxlag = maxlag
        self.lambda_ = lambda_
        self.K = K
        self.seas_method = seas_method
        self.trend_method = trend_method
        self.nrounds_method = nrounds_method
        self.nrounds_max = nrounds_max
        self.nfold = nfold
        self.features_eliminator_builder = features_eliminator_builder
        self.kwargs = kwargs
        self.trained_model_params = None

    @property
    def summary(self):
        """
        Get model summary
        :return: summary info
        :rtype: str
        """
        assert self.trained_model_params is not None
        return summary(self.trained_model_params)

    def fit(self, y, frequency, xreg=None, verbose=True):
        """
        Train model
        :param y: time series to forecast
        :type y: pd.Series|np.ndarray
        :param frequency: frequency
        :type frequency: int
        :param xreg: external values
        :type xreg: pd.DataFrame|np.ndarray
        :param verbose: verbose train process?
        :type verbose: bool
        :return: trained model
        :rtype: ForecastXGB
        """
        self.trained_model_params = xgbar(y, frequency, xreg=xreg,
                                          maxlag=self.maxlag, lambda_=self.lambda_, K=self.K,
                                          seas_method=self.seas_method,
                                          trend_method=self.trend_method,
                                          nrounds_method=self.nrounds_method,
                                          nfold=self.nfold,
                                          verbose=verbose,
                                          nrounds_max=self.nrounds_max,
                                          features_eliminator_builder=self.features_eliminator_builder,
                                          **self.kwargs)
        return self

    def forecast(self, length=None, xreg=None):
        """
        Make prediction
        :param length: length of prediction (if xreg not presented)
        :type length: int
        :param xreg: external variables
        :type xreg: pd.DataFrame|np.ndarray
        :return: prediction (1d-array)
        :rtype: np.ndarray
        """
        assert self.trained_model_params is not None
        return forecast(self.trained_model_params, length, xreg)
