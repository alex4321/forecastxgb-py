from unittest import TestCase
import numpy as np
from .datasets import gas, gas_frequency, usconsumption, usconsumption_frequency
from .forecastxgb import ForecastXGB
from sklearn.metrics import mean_squared_error
import pandas as pd
import warnings
from sklearn.feature_selection import RFECV
from xgboost import XGBRegressor
warnings.filterwarnings('ignore')


def root_mean_squared_scaled_error(y_true, y_pred):
    error = mean_squared_error(y_true, y_pred)
    return np.sqrt(error) / np.median(y_true)


def spearman_corr(y_true, y_pred):
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred
    }).corr(method='spearman')
    return df['y_true']['y_pred']


class ForecastXGBTest(TestCase):
    def test_autoregression(self):
        model = ForecastXGB().fit(gas, gas_frequency, verbose=False)
        print(model.summary)
        fc = model.forecast(length=10)
        right = [50697.04, 47071.73, 51041.30, 37013.81, 42412.93,
                 44348.35, 49622.32, 54831.63, 59418.39, 61656.44]
        self.assertLess(root_mean_squared_scaled_error(right, fc), 0.01)

    @staticmethod
    def _features_eliminator_builder(num_boost_round, **kwargs):
        return RFECV(XGBRegressor(n_estimators=num_boost_round, **kwargs),
                     scoring='neg_mean_squared_error')

    def test_autoregression_with_RFE(self):
        model = ForecastXGB(features_eliminator_builder=ForecastXGBTest._features_eliminator_builder)\
                .fit(gas, gas_frequency, verbose=False)
        print(model.summary)
        fc = model.forecast(length=10)
        right = [50697.04, 47071.73, 51041.30, 37013.81, 42412.93,
                 44348.35, 49622.32, 54831.63, 59418.39, 61656.44]
        self.assertLess(root_mean_squared_scaled_error(right, fc), 0.1)

    def test_xreg(self):
        consumption = usconsumption['consumption']
        income = usconsumption[['income']]
        consumption_model = ForecastXGB().fit(consumption, usconsumption_frequency, income)
        income_model = ForecastXGB().fit(income['income'], usconsumption_frequency)
        income_future = pd.DataFrame({
            'income': income_model.forecast(length=10)
        })
        income_future_right = [0.5731711, 0.5229242, 1.0045462, 0.1340497, 0.1340497,
                               0.5877231, 0.7185796, 0.5213639, 0.5877231, 0.5877231]
        self.assertLess(root_mean_squared_scaled_error(income_future_right, income_future), 0.01)
        fc = consumption_model.forecast(xreg=income_future)
        fc_right = [0.49649048, 0.29876488, 0.30465472, -0.55017745, -0.28470188,
                    0.40086550, 0.03414938, 0.59014469,  0.59604609, 0.92111415]
        # This model isn't stable enough to predict exactly value of consumption
        # but give good spearman corellation
        self.assertGreater(spearman_corr(fc_right, fc), 0.4)
