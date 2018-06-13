import numpy as np
import pandas as pd


def _np_nan_wrapper(func):
    def _wrapper(values):
        result = np.zeros(values.shape)
        nan_idx = np.isnan(values)
        result[nan_idx] = float('NaN')
        not_nan_idx = np.logical_not(nan_idx)
        result[not_nan_idx] = func(values[not_nan_idx])
        if isinstance(values, pd.Series):
            result_series = pd.Series(result)
            result_series.index = values.index
            return result_series
        return result

    return _wrapper


_sign = _np_nan_wrapper(np.sign)
_abs = _np_nan_wrapper(np.abs)
_log = _np_nan_wrapper(np.log)
_exp = _np_nan_wrapper(np.exp)


def jd_mod(y, lambda_):
    """
    function to perform transformation as per John and Draper's
        "An Alternative Family of Transformations"
    John and Draper's modulus transformation
    :param y: series
    :type y: np.ndarray|pd.Series
    :param lambda_: lambda value
    :type lambda_: float
    :return: np.ndarray|pd.Series
    """
    if lambda_ != 0:
        yt = _sign(y) * (((_abs(y) + 1) ** lambda_ - 1) / lambda_)
    else:
        yt = _sign(y) * (_log(_abs(y) + 1))
    return yt


def inv_jd_mod(yt, lambda_):
    """
    function to reverse transformation as per John and Draper's
        "An Alternative Family of Transformations"
    John and Draper's modulus transformation
    :param y: series
    :type y: np.ndarray|pd.Series
    :param lambda_: lambda value
    :type lambda_: float
    :return: np.ndarray|pd.Series
    """
    if lambda_ != 0:
        y = ((_abs(yt) * lambda_ + 1) ** (1 / lambda_) - 1) * _sign(yt)
    else:
        y = (_exp(_abs(yt)) - 1) * _sign(yt)
    return y


def lagv(x, maxlag, varname):
    """
    Function to create matrix of lagged versions of variable
    :param x: source series
    :type x: np.ndarray|pd.Series
    :param maxlag: max lag value
    :type maxlag: int
    :param varname: variable name (to set esult dataframe collumns)
    :type varname: str
    :return: dataframe with lagged versions of variable
    :rtype: pd.DataFrame
    """
    x_ = np.array(x)
    nrows = len(x_) - maxlag
    z = np.zeros([nrows, maxlag])
    for i in range(maxlag):
        lagged_column = x_[maxlag - i - 1:][:nrows]
        z[:, i] = lagged_column
    columns = ['{0}_lag_{1}'.format(varname, i + 1)
               for i in range(maxlag)]
    return z, columns


def lagvm(x, maxlag):
    """
    Function to create lagged versions of external variables
    :param x: source dataframe
    :type x: pd.DataFrame
    :param maxlag: max lag value
    :type maxlag: int
    :return: dataframe with lagged version of variables - (INCLUDE ZERO LAG!)
    :rtype: pd.DataFrame
    """
    xlagged = [np.array(x)[maxlag:, :]]
    columns_lagged = ['{0}_lag_0'.format(column) for column in x.columns]
    for column in x.columns:
        varlagged, columns = lagv(np.array(x[column]), maxlag, column)
        xlagged.append(varlagged)
        columns_lagged += columns
    result = np.hstack(xlagged)
    return result, columns_lagged
