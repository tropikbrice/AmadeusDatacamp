import pandas as pd
import os
import numpy as np
import pandas as pd

class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y_array):
        pass

    def transform(self, X_df):
        X_encoded = X_df
        path = os.path.dirname(__file__)
        data = pd.read_csv(os.path.join(path, "data_amadeus.csv"))
        
        X_encoded = data

        ###-------------------
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Departure'], prefix='d'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Arrival'], prefix='a'))
        X_encoded = X_encoded.drop('Departure', axis=1)
        X_encoded = X_encoded.drop('Arrival', axis=1)

        # following http://stackoverflow.com/questions/16453644/regression-with-date-variable-using-scikit-learn
        X_encoded['DateOfDeparture'] = pd.to_datetime(X_encoded['DateOfDeparture'])
        X_encoded['year'] = X_encoded['DateOfDeparture'].dt.year
        #X_encoded['month'] = X_encoded['DateOfDeparture'].dt.month
        X_encoded['day'] = X_encoded['DateOfDeparture'].dt.day
        X_encoded['weekday'] = X_encoded['DateOfDeparture'].dt.weekday
        X_encoded['week'] = X_encoded['DateOfDeparture'].dt.week
        X_encoded['n_days'] = X_encoded['DateOfDeparture'].apply(lambda date: (date - pd.to_datetime("1970-01-01")).days)

        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['year'], prefix='y'))
        #X_encoded = X_encoded.join(pd.get_dummies(X_encoded['month'], prefix='m'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['day'], prefix='d'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['weekday'], prefix='wd'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['week'], prefix='w'))

        ### Je vire les variables de classse
        X_encoded = X_encoded.drop('year', axis=1)
        #X_encoded = X_encoded.drop('month', axis=1)
        X_encoded = X_encoded.drop('day', axis=1)
        X_encoded = X_encoded.drop('weekday', axis=1)
        X_encoded = X_encoded.drop('week', axis=1)
        X_encoded = X_encoded.drop('n_days', axis=1)
        ###------

        X_array = X_encoded.values
        return X_array
