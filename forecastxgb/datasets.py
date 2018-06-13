from dateutil.relativedelta import relativedelta
import json
import os
import pandas as pd


def _read_gas():
    fname = os.path.join(os.path.dirname(__file__), 'gas.json')
    with open(fname, 'r') as src:
        data = json.load(src)
    from_date = pd.to_datetime(data['from'])
    to_date = pd.to_datetime(data['to'])
    dates = pd.date_range(from_date, to_date, freq='MS')
    df = pd.DataFrame({
        'date': dates,
        'value': data['values']
    })
    df.set_index('date', inplace=True)
    return df['value'], 12


def _read_usconsumption():
    fname = os.path.join(os.path.dirname(__file__), 'usconsumption.csv')
    df = pd.read_csv(fname)
    return df.set_index(['year', 'quarter']), 4


gas, gas_frequency = _read_gas()
usconsumption, usconsumption_frequency = _read_usconsumption()