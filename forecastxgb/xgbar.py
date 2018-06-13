import pandas as pd
import warnings
import xgboost as xgb
from .utils import *

#region Arguments checking

def _check_numeric_array(arr_, pandas_accepatable_type, dimensions, message):
    if (not isinstance(arr_, np.ndarray)) and (not isinstance(arr_, pandas_accepatable_type)):
        raise ValueError(message)
    arr = np.array(arr_)
    if arr.dtype not in [np.float16, np.float32, np.float64,
                         np.int8, np.int16, np.int32, np.int64,
                         np.uint8, np.uint16, np.uint32, np.uint64]:
        raise ValueError(message)
    if len(arr.shape) != dimensions:
        raise ValueError(message)


def _check_kwargs(**kwargs):
    if 'silent' not in kwargs:
        kwargs['silent'] = True
    if 'obj' not in kwargs:
        kwargs['obj'] = 'reg:linear'
    return kwargs


def _check_y(y):
    _check_numeric_array(y, pd.Series, 1, 'y must be a 1-d numeric array')
    return pd.Series(y)


def _check_xreg(y, xreg):
    if xreg is not None:
        _check_numeric_array(xreg, pd.DataFrame, 2, 'xreg must be a 2-d numeric array')
        assert len(y) == len(xreg), 'xreg must be same length as y'
    if xreg is not None and isinstance(xreg, np.ndarray):
        xreg = pd.DataFrame(xreg)
        xreg.columns = ['feature_{0}'.format(i)
                        for i in range(xreg.shape[1])]
    return xreg


def _check_maxlag(y, maxlag, frequency, seas_method):
    if maxlag is None:
        maxlag = max(8, 2 * frequency)
    if maxlag < frequency and seas_method == "dummies":
        raise ValueError("At least one full period of lags needed when seas_method = dummies.")
    max_possible_lag = len(y) - frequency - round(frequency / 4.0)
    if maxlag > max_possible_lag:
        warnings.warn("y is too short for {0} to be the value of maxlag.  Reducing maxlags to {1} instead.".format(
            maxlag,
            max_possible_lag
        ))
        maxlag = max_possible_lag
    return maxlag


def _check_K(K, frequency):
    if K is None:
        K = max(1, min(round(frequency / 4 - 1), 10))
    return K


def _check_nfold(y, nfold):
    if nfold is None:
        if len(y) > 30:
            nfold = 10
        else:
            nfold = 5
    return nfold


def _check_nrounds_method(y, maxlag, nrounds_method):
    dataset_size = len(y) - maxlag
    if nrounds_method == 'cv' and dataset_size < 15:
        warnings.warn("y is too short for cross-validation.  Will validate on the most recent 20 per cent instead.")
        nrounds_method = 'v'
    return nrounds_method


def _check_args(y, xreg, maxlag, K, frequency, seas_method, nfold, nrounds_method, **kwargs):
    if len(y) < 4:
        raise ValueError("Too short. I need at least four observations.")
    kwargs = _check_kwargs(**kwargs)
    y = _check_y(y)
    xreg = _check_xreg(y, xreg)
    maxlag = _check_maxlag(y, maxlag, frequency, seas_method)
    K = _check_K(K, frequency)
    nfold = _check_nfold(y, nfold)
    nrounds_method = _check_nrounds_method(y, maxlag, nrounds_method)
    return y, xreg, maxlag, K, nfold, nrounds_method, kwargs

#endregion


#region Feature building (dealing with trends/seasonality/lags)

def _build_features(y, xreg, frequency, K, lambda_, seas_method, trend_method, maxlag):
    y = jd_mod(y, lambda_)
    if seas_method == 'decompose':
        raise NotImplementedError()
        # decomp <- decompose(y, type = "multiplicative")
        # y <- seasadj(decomp)
    trend_diffs_count = 0
    if trend_method == 'differencing':
        raise NotImplementedError()
        # alpha = 0.05
        # dodiff <- TRUE
        # while(dodiff){
        #   suppressWarnings(dodiff <- tseries::kpss.test(origy)$p.value < alpha)
        #   if(dodiff){
        #     trend_diffs_count <- trend_diffs_count + 1
        #     origy <- ts(c(0, diff(origy)), start = start(origy), frequency = f)
        #   }
        # }

    dataset_size = len(y) - maxlag
    y_target = y.tail(len(y) - maxlag)
    if seas_method == 'dummies' and frequency > 1:
        ncol_x = maxlag + frequency - 1
    elif seas_method == 'decompose':
        ncol_x = maxlag
    elif seas_method == 'fourier' and frequency > 1:
        ncol_x = maxlag + K * 2
    elif seas_method == 'none':
        ncol_x = maxlag
    else:
        raise NotImplementedError()

    x = np.zeros([dataset_size, ncol_x + 1])
    x[:, :maxlag], columns = lagv(y, maxlag, 'y')

    if frequency > 1 and seas_method == 'dummies':
        tmp = np.arange(dataset_size) % frequency
        seasons = np.zeros([len(tmp), frequency])
        season_columns = []
        for i, season in enumerate(sorted(set(tmp))):
            season_columns += ['season_{0}'.format(i + 1)]
            seasons[tmp == season, i] = 1.0
        x[:, maxlag : maxlag + frequency + 1] = seasons
        columns += season_columns
    if frequency > 1 and seas_method == 'fourier':
        raise NotImplementedError()
        # fx <- fourier(y2, K = K)
        # x[ , (maxlag + 1):ncolx] <- fx
        # colnames(x) <- c(paste0("lag", 1:maxlag), colnames(fx))
    x = pd.DataFrame(x)
    x.columns = columns

    if xreg is not None:
        xreg_lagged, xreg_columns = lagvm(xreg, maxlag)
        for i, column in enumerate(xreg_columns):
            x[column] = xreg_lagged[:, i]

    return x, y_target, trend_diffs_count

#endregion


# region High-level forecasting model train functionality

def _get_best_nrounds(nrounds_method, X, y_target, nfold, nrounds, verbose, **kwargs):
    if nrounds_method == 'cv':
        cv = xgb.cv(params=kwargs, num_boost_round=nrounds, dtrain=xgb.DMatrix(X, y_target), nfold=nfold, early_stopping_rounds=5, verbose_eval=verbose)
        cv_scores = np.array(cv[list(cv.columns)[2]])
        nrounds_use = cv_scores.argmin() + 1
    elif nrounds_method == 'v':
        raise NotImplementedError()
        #nrounds_use < - validate_xgbar(y, xreg = xreg, ...) $best_nrounds
    else:
        nrounds_use = nrounds
    return nrounds_use


def _fitted_reverse_transformations(fitted, trend_method, seas_method, lambda_):
    # back transform the differencing
    if trend_method == 'differencing':
        raise NotImplementedError()
        #for (i in 1:diffs){
        #fitted[! is.na(fitted)] < - ts(cumsum(fitted[! is.na(fitted)]), start = start(origy), frequency = f)
        #}
        #fitted < - fitted + JDMod(untransformedy[maxlag + 1], lambda = lambda )
        #}
    # back transform the seasonal adjustment:
    if seas_method == 'decompose':
        raise NotImplementedError()
        #fitted < - fitted * decomp$seasonal
    fitted = inv_jd_mod(fitted, lambda_)
    return fitted


def xgbar(y, frequency, xreg=None, maxlag=None, lambda_=1, K=None, seas_method='dummies',
          trend_method='none', nrounds_method='cv', nfold=None, verbose=True, nrounds_max=100,
          features_eliminator_builder=None,
          **kwargs):
    # validate arguments, apply conversions if need
    y, xreg, maxlag, K, nfold, nrounds_method, kwargs = _check_args(y, xreg, maxlag, K,
                                                                    frequency,
                                                                    seas_method,
                                                                    nfold,
                                                                    nrounds_method,
                                                                    **kwargs)
    untransformed_y = np.copy(np.array(y))
    if xreg is not None:
        orig_xreg = xreg.copy()
    else:
        orig_xreg = None
    # Apply transformations (e.g. remove trends, seasonality) to build features
    X, y_target, trend_diffs_count = _build_features(y, xreg, frequency, K, lambda_, seas_method, trend_method, maxlag)
    # Choose params and train model
    nrounds_use = _get_best_nrounds(nrounds_method, X, y_target, nfold, nrounds_max, verbose, **kwargs)
    if features_eliminator_builder:
        features_eliminator = features_eliminator_builder(num_boost_round=nrounds_use, **kwargs)
        features_eliminator.fit(X, y_target)
        chosen_features = features_eliminator.get_support()
    else:
        chosen_features = np.array(range(len(X.columns)))
    model = xgb.train(params=kwargs, num_boost_round=nrounds_use,
                      dtrain=xgb.DMatrix(X[np.array(X.columns)[chosen_features]], y_target),
                      verbose_eval=verbose)
    return {
        'maxlag': maxlag,
        'y': untransformed_y,
        'y_target': y_target,
        'x': X,
        'chosen_features': chosen_features,
        'model': model,
        'diffs': trend_diffs_count,
        'lambda_': lambda_,
        'frequency': frequency,
        'seas_method': seas_method,
        'orig_xreg': orig_xreg,
    }

# endregion


# region Forecasting functionality (for trained model)

def _get_xreg_info(xreg, h, params):
    if xreg is not None:
        assert xreg.shape[1] == params['orig_xreg'].shape[1], \
            'Number of columns in xreg doesn\'t match the original xgbar object.'
        if h:
            warnings.warn("Ignoring h and forecasting {0} periods from xreg.".format(len(xreg)))
        h = len(xreg)
        xreg_all_info = pd.DataFrame(np.vstack([params['orig_xreg'], xreg]))
        xreg_all_info.columns = params['orig_xreg'].columns
        xreg_all_info_lagged, xreg_lagged_columns = lagvm(xreg_all_info, params['maxlag'])
        xreg_all_info_lagged_df = pd.DataFrame(xreg_all_info_lagged)
        xreg_all_info_lagged_df.columns = xreg_lagged_columns
        xreg_lagged_interested = xreg_all_info_lagged_df.tail(h).copy().reset_index()[xreg_lagged_columns]
    else:
        xreg_lagged_interested = None
    return xreg_lagged_interested, h


def _forecast_next_value(x, y, xregpred, i, params, frequency, seas_method, chosen_features):
    if xregpred is not None:
        xregpred = np.array(xregpred)
    newrow = np.array([y[-1]] + list(x[-1, :params['maxlag']-1]))
    if params['maxlag'] == 1:
        newrow = newrow[[-1]]
    # seasonal dummies if 'dummies':
    if frequency > 1 and seas_method == "dummies":
        newrow = np.array(list(newrow) + list(x[-frequency, params['maxlag'] : params['maxlag'] + frequency]))
    # seasonal dummies if 'fourier':
    if frequency > 1 and seas_method == 'fourier':
        # for fourier variables,
        #newrow < - c(newrow, fxh[i,])
        raise NotImplementedError()
    if xregpred is not None:
        newrow = np.array(list(newrow) + list(xregpred))
    newrow = np.array([newrow])
    newrow_df = pd.DataFrame(newrow)
    newrow_df.columns = params['x'].columns
    newrow_dmatrix = xgb.DMatrix(newrow_df[np.array(newrow_df.columns)[chosen_features]])
    pred = params['model'].predict(newrow_dmatrix)
    x = np.vstack((x, newrow))
    y = np.array(list(y) + list(pred))
    return x, y


def forecast(params, h=None, xreg=None):
    # Build features (lagged values/seasons/other like it)
    xreg_lagged_interested, h = _get_xreg_info(xreg, h, params)
    if h is None:
        if params['frequency'] > 1:
            h = 2 * params['frequency']
        else:
            h = 10
        warnings.warn("No h provided so forecasting forward {0} periods.".format(h))

    frequency = params['frequency']
    lambda_ = params['lambda_']
    seas_method = params['seas_method']
    chosen_features = params['chosen_features']

    # forecast fourier pairs
    if frequency > 1 and seas_method == "fourier":
        raise NotImplementedError()
        #fxh < - fourier(object$y2, K = object$K, h = h)

    # Forecast inner-result (before restoring trends/seasons)
    x = np.array(params['x'])
    y = np.array(params['y_target'])
    for i in range(h):
        if xreg_lagged_interested is None:
            xregpred = None
        else:
            xregpred = xreg_lagged_interested.iloc[i, :]
        x, y = _forecast_next_value(x, y, xregpred, i, params, frequency, seas_method, chosen_features)

    # Restore trends
    if params['diffs'] > 0:
        raise NotImplementedError()
        #for(i in 1:object$diffs){
        #  y <- ts(cumsum(y)  , start = start(y), frequency = f)
        #}
        #y <- y + JDMod(object$y[length(object$y)], lambda = lambda)
    # Restore seasons
    if seas_method == "decompose":
        raise NotImplementedError()
        #multipliers <- utils::tail(object$decomp$seasonal, f)
        #if(h < f){
        #    multipliers <- multipliers[1:h]
        #}
        #y <- y * as.vector(multipliers)

    y = inv_jd_mod(y, lambda_=lambda_)
    return y[-h:]

#endregion


def summary(params):
    result = 'Importance of features in the xgboost model:\n'
    model = params['model']
    scores = model.get_score(importance_type='gain')
    features_sorted = sorted(scores.keys(), key=lambda feature: -scores[feature])
    for feature in features_sorted:
        result += '{0}\t{1}\n'.format(feature, scores[feature])
    return result.strip()
