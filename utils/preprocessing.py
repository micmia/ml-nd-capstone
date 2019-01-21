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
    # elif dtype == 'test':
    #     data['Open'].fillna(1, inplace=True)

    # Split Date column into Year, Month, Day, DayOfWeek, DayOfYear, WeekOfYear
    # Based on https://www.kaggle.com/cast42/xgboost-in-python-with-rmspe-v2
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    data['DayOfYear'] = data['Date'].dt.dayofyear
    data['WeekOfYear'] = data['Date'].dt.weekofyear

    data_school_holidays = data.groupby(['Store', 'Year', 'WeekOfYear'])['SchoolHoliday'].sum().reset_index(
        name='SchoolHolidaysThisWeek')
    data_school_holidays['SchoolHolidaysLastWeek'] = data_school_holidays['SchoolHolidaysThisWeek'].shift(-1)
    data_school_holidays['SchoolHolidaysNextWeek'] = data_school_holidays['SchoolHolidaysThisWeek'].shift()
    data = data.merge(data_school_holidays, on=['Store', 'Year', 'WeekOfYear'], how='left', validate='m:1')

    mappings = {'0': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4}
    data['StoreType'].replace(mappings, inplace=True)
    data['Assortment'].replace(mappings, inplace=True)
    data['StateHoliday'].replace(mappings, inplace=True)

    data['CompetitionDistance'] = np.log1p(data['CompetitionDistance'].fillna(data['CompetitionDistance'].mean()))
    data['CompetitionOpen'] = 12 * (data['Year'] - data['CompetitionOpenSinceYear']) + (
        data['Month'] - data['CompetitionOpenSinceMonth'])

    data['Promo2Open'] = 12 * (data['Year'] - data['Promo2SinceYear']) + (data['WeekOfYear'] - data[
        'Promo2SinceWeek']) / 4.0
    data['Promo2Open'] = data['Promo2Open'].apply(lambda x: x if x > 0 else 0)

    data['PromoInterval'] = data['PromoInterval'].fillna('')
    data['InPromoMonth'] = data.apply(
        lambda x: 1 if (x['Date'].strftime('%b') if not x['Date'].strftime('%b') == 'Sep' else 'Sept') in x[
            'PromoInterval'].split(',') else 0, axis=1)

    if dtype == 'train':
        data.drop([
            'Date',
            'Customers',
            'Promo2SinceWeek',
            'Promo2SinceYear',
            'PromoInterval'],
            axis=1, inplace=True)
    elif dtype == 'test':
        data.drop([
            'Date',
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
