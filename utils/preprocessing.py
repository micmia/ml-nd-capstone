import pandas as pd
import numpy as np


def read_csv(files):
    data_train = pd.read_csv(files[0],
                             dtype={
                                 'StateHoliday': 'category',
                                 'SchoolHoliday': 'int'},
                             parse_dates=['Date'])

    data_test = pd.read_csv(files[1],
                            dtype={
                                'StateHoliday': 'category',
                                'SchoolHoliday': 'int'},
                            parse_dates=['Date'])

    data_store = pd.read_csv(files[2],
                             dtype={
                                 'StoreType': 'category',
                                 'Assortment': 'category',
                                 'CompetitionOpenSinceMonth': float,
                                 'CompetitionOpenSinceYear': float,
                                 'Promo2': float,
                                 'Promo2SinceWeek': float,
                                 'Promo2SinceYear': float,
                                 'PromoInterval': str})

    return data_train, data_test, data_store


def combine(data, data_store):
    return pd.merge(data, data_store, on='Store', how='left')


def transform(data, dtype):
    data = data.copy()

    if dtype == 'train':
        # Use only Sales > 0 and Open != 0 to simplify calculation of RMSPE
        # RMSPE does not take into account Sales = 0
        data = data[(data['Sales'] > 0) & (data['Open'] != 0)]
        data = data.sort_values(by='Date')

        # Remove outliers of Sales and CompetitionDistance
        # data = data[~(np.abs(data['Sales'] - data['Sales'].mean()) > (4.2 * data['Sales'].std()))]

    # Split Date column into Year, Month, Day, DayOfWeek, DayOfYear, WeekOfYear
    # Based on https://www.kaggle.com/cast42/xgboost-in-python-with-rmspe-v2
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    data['DayOfYear'] = data['Date'].dt.dayofyear
    data['WeekOfYear'] = data['Date'].dt.weekofyear

    # data_state_holiday = pd.get_dummies(data['StateHoliday'], prefix='StateHoliday')
    data_store_type = pd.get_dummies(data['StoreType'], prefix='StoreType')
    data_assortment = pd.get_dummies(data['Assortment'], prefix='Assortment')

    data['CompetitionDistance'] = np.log1p(data['CompetitionDistance'].fillna(data['CompetitionDistance'].median()))
    data['CompetitionOpen'] = 12 * (data['Year'] - data['CompetitionOpenSinceYear']) + (
        data['Month'] - data['CompetitionOpenSinceMonth'])
    data['CompetitionOpen'] = data['CompetitionOpen'].apply(lambda x: x if x > 0 else 0)

    data['Promo2Open'] = 12 * (data['Year'] - data['Promo2SinceYear']) + (data['WeekOfYear'] - data[
        'Promo2SinceWeek']) / 4.0
    data['Promo2Open'] = data['Promo2Open'].apply(lambda x: x if x > 0 else 0)

    data['PromoInterval'] = data['PromoInterval'].fillna('')
    data['InPromoMonth'] = data.apply(
        lambda x: 1 if (x['Date'].strftime('%b') if not x['Date'].strftime('%b') == 'Sep' else 'Sept') in x[
            'PromoInterval'].split(',') else 0, axis=1)

    data = pd.concat(
        [data, data_store_type, data_assortment], axis=1)

    if dtype == 'train':
        data.drop([
            'Date',
            'Customers',
            'StateHoliday',
            'StoreType',
            'Assortment',
            'Promo2SinceWeek',
            'Promo2SinceYear',
            'PromoInterval'],
            axis=1, inplace=True)
    elif dtype == 'test':
        data['StateHoliday_b'] = 0
        data['StateHoliday_c'] = 0
        data.drop([
            'Date',
            'StateHoliday',
            'StoreType',
            'Assortment',
            'Promo2SinceWeek',
            'Promo2SinceYear',
            'PromoInterval'],
            axis=1, inplace=True)

    if dtype == 'train':
        return data.drop(['Sales'], axis=1), data['Sales']
    elif dtype == 'test':
        return data.drop(['Id'], axis=1)


def transform_y(y):
    return np.log1p(y)


def restore_y(y):
    return np.expm1(y)
