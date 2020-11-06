import logging
logger = logging.getLogger(__name__)

from scipy import stats
import datetime as dt
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

class Preprocess:
    ''' Input: raw data
        Output: reduced, clean data
    '''

    def __init__(self, data):
        self.data = data

    def reduce_data(self, select_columns, target_column, classification):
        ''' Select rows and columns '''
        self.select_columns = select_columns
        self.data = self.data[self.select_columns]
        self.data = self.data.loc[self.data[target_column].isin(classification)]
        logging.info('Dimension of reduced data is %i x %i' % (self.data.shape[0], self.data.shape[1]))
        for c in classification:
            logging.info('Classification: %s has %i rows' % (c, len(self.data[self.data[target_column] == c])))
        return self.data

    def make_catcols(self, categorical_columns):
        ''' Change some columns to categorical and do one-hot-encoding '''
        self.data[categorical_columns] = self.data[categorical_columns].astype('category')
        for cat_col in categorical_columns:
            assert cat_col in self.select_columns
            logging.info('Categorical column: %s' % cat_col)
            ohe_col   = pd.get_dummies(self.data[cat_col], prefix=cat_col, dummy_na=True)
            self.data = pd.concat([self.data, ohe_col], axis=1)
            self.data = self.data.drop(cat_col, axis=1)
        return self.data

    def make_ordcols(self, ordered_columns):
        ''' Change ordered columns to numbers '''
        for col in ordered_columns:
            assert col in self.select_columns
            logging.info('Ordered column: %s' % col)
            self.data[col] = LabelEncoder().fit_transform(self.data[col])
        return self.data

    def str_to_num(self, colname, suffix, replace):
        ''' Change string to number '''
        assert colname in self.select_columns
        self.data[colname] = self.data[colname].astype('str')
        self.data[colname] = self.data[colname].map(lambda x: x.rstrip(suffix))
        for key in replace.keys():
            self.data.loc[self.data[colname] == key, colname] = replace[key]
            logging.info('Replacing {} with {} in {}'.format(key, replace[key], colname))
        self.data[colname] = self.data[colname].astype('float')
        logging.info('Minimum {} = {} {}'.format(colname, self.data[colname].min(), suffix))
        logging.info('Maximum {} = {} {}'.format(colname, self.data[colname].max(), suffix))

    def calc_time_interval(self, time1, time2, time_diff):
        ''' time_diff = time2 - time1 (in days) '''
        assert (time1 in self.select_columns) and (time2 in self.select_columns)
        # Make datetime format
        date_columns = [time1, time2]
        self.data[date_columns] = self.data[date_columns].apply(pd.to_datetime)
        for col in date_columns: assert self.data[col].dtypes == '<M8[ns]'
        # Convert into time interval and then to float that count days
        self.data[time_diff] = (self.data[time2] - self.data[time1]).dt.days
        self.data.drop(date_columns, axis=1, inplace=True)
        return self.data

    def impute_nan(self, impute):
        ''' Raplacing NaN values '''
        na_dict = {}
        for col in self.data.columns:
            if self.data[col].isna().sum() > 0:
                if impute == 'median': na_dict[col] = np.nanmedian(self.data[col])
                elif impute == 'mean': na_dict[col] = np.nanmean(self.data[col])
                else: na_dict[col] = stats.mode(self.data[col])[0][0]
                logging.info('Imputing %s with %.2f' % (col, na_dict[col]))
        self.data = self.data.fillna(value=na_dict)
        return self.data
