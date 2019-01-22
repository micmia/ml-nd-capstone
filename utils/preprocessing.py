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


def transform(data_train, data_test):
    data_train_ = data_train.copy()
    data_test_ = data_test.copy()

    data_train_ = data_train_[(data_train_['Sales'] > 0) & (data_train_['Open'] != 0)]
    data_test_['Open'] = data_test_['Open'].fillna(1)

    def process(data):
        data['Year'] = data['Date'].dt.year
        data['Month'] = data['Date'].dt.month
        data['Day'] = data['Date'].dt.day
        data['DayOfWeek'] = data['Date'].dt.dayofweek
        data['DayOfYear'] = data['Date'].dt.dayofyear
        data['WeekOfYear'] = data['Date'].dt.weekofyear
        data['Quarter'] = (data['Date'].dt.month - 1) // 3 + 1

        mappings = {'0': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4}
        data['StoreType'].replace(mappings, inplace=True)
        data['Assortment'].replace(mappings, inplace=True)
        data['StateHoliday'].replace(mappings, inplace=True)

        data['CompetitionDistance'] = data['CompetitionDistance'].fillna(data['CompetitionDistance'].median())

        # Extend features
        data['CompetitionOpen'] = 12 * (data['Year'] - data['CompetitionOpenSinceYear']) + (
                data['Month'] - data['CompetitionOpenSinceMonth']).apply(lambda x: x if x > 0 else 0)

        data['Promo2Open'] = 12 * (data['Year'] - data['Promo2SinceYear']) + (data['WeekOfYear'] - data[
            'Promo2SinceWeek']) / 4.0
        data['Promo2Open'] = data['Promo2Open'].apply(lambda x: x if x > 0 else 0)

        data['PromoInterval'] = data['PromoInterval'].fillna('')
        data['InPromoMonth'] = data.apply(
            lambda x: 1 if (x['Date'].strftime('%b') if not x['Date'].strftime('%b') == 'Sep' else 'Sept') in x[
                'PromoInterval'].split(',') else 0, axis=1)

        # data_school_holidays = data.groupby(['Store', 'Year', 'WeekOfYear'])['SchoolHoliday'].sum().reset_index(
        #     name='SchoolHolidaysThisWeek')
        # data_school_holidays['SchoolHolidaysLastWeek'] = data_school_holidays['SchoolHolidaysThisWeek'].shift(-1)
        # data_school_holidays['SchoolHolidaysNextWeek'] = data_school_holidays['SchoolHolidaysThisWeek'].shift()
        # data_school_holidays.fillna(0)
        # data = data.merge(data_school_holidays, on=['Store', 'Year', 'WeekOfYear'], how='left', validate='m:1')

        data_meanlog_salesbystore = data_train_.groupby(['Store'])['Sales'].mean().reset_index(
            name='MeanLogSalesByStore')
        data_meanlog_salesbystore['MeanLogSalesByStore'] = np.log1p(data_meanlog_salesbystore['MeanLogSalesByStore'])
        data = data.merge(data_meanlog_salesbystore, on=['Store'], how='left', validate='m:1')

        data_meanlog_salesbydow = data_train_.groupby(['DayOfWeek'])['Sales'].mean().reset_index(
            name='MeanLogSalesByDOW')
        data_meanlog_salesbydow['MeanLogSalesByDOW'] = np.log1p(data_meanlog_salesbystore['MeanLogSalesByStore'])
        data = data.merge(data_meanlog_salesbydow, on=['DayOfWeek'], how='left', validate='m:1')

        data_meanlog_salesbymonth = data_train_.groupby(['Month'])['Sales'].mean().reset_index(
            name='MeanLogSalesByMonth')
        data_meanlog_salesbymonth['MeanLogSalesByMonth'] = np.log1p(data_meanlog_salesbymonth['MeanLogSalesByMonth'])
        data = data.merge(data_meanlog_salesbymonth, on=['Month'], how='left', validate='m:1')

        return data

    features = [
        'Store', 'DayOfWeek', 'Year', 'Month', 'Day', 'DayOfYear', 'WeekOfYear', 'Quarter', 'Open', 'Promo',
        'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
        'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'CompetitionOpen',
        'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'Promo2Open', 'InPromoMonth',
        'MeanLogSalesByStore', 'MeanLogSalesByDOW', 'MeanLogSalesByMonth']

    data_train_ = process(data_train_)
    data_test_ = process(data_test_)

    return (data_train_[features], np.log1p(data_train_['Sales'])), data_test_[features]
