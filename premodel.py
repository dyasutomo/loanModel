import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from joblib import dump, load

class Premodel:
    ''' Input: cleaned data
        Output: training and test set
        options: upsampling and scaling data
    '''

    def __init__(self, data, target_column=None):
        if target_column == None:
            self.X = data
        else:
            self.y = data[target_column]
            self.X = data.drop(target_column, axis=1)
            self.target_column = target_column
            logging.info('number of classes: %i' % len(np.unique(self.y)))
            assert len(np.unique(self.y)) >= 2

    def upsampling(self):
        ''' Upsampling to make balanced sample between default and paid loan '''
        logging.info('Before upsampling: %i rows' % len(self.y))
        self.X, self.y = SMOTE().fit_resample(self.X, self.y)
        logging.info('After upsampling: %i rows' % len(self.y))
        return self.X, self.y

    def splitting(self, testsize):
        ''' Splitting into train and test sets '''
        self.X = pd.DataFrame(self.X, columns=self.X.columns)
        self.y = pd.DataFrame(self.y, columns=[self.target_column])
        logging.info('Splitting into train and test with test size = %.2f' % testsize)
        self.X, self.X_test, self.y, self.y_test = train_test_split(self.X, self.y,
                                                                                test_size=testsize,
                                                                                random_state=34)
        return self.X, self.X_test, self.y, self.y_test

    def scaling_fit_transform(self, numerical_columns, scaler_file='std_scaler.bin'):
        ''' Scaling data '''
        scaler = StandardScaler()
        self.X[numerical_columns] = scaler.fit_transform(self.X[numerical_columns])
        logging.info('Scaling numerical columns ...')
        dump(scaler, scaler_file, compress=True)
        return self.X

    def scaling_transform(self, numerical_columns, scaler_file='std_scaler.bin'):
        scaler = load(scaler_file)
        cols   = self.X.columns
        self.X[numerical_columns] = scaler.transform(self.X[numerical_columns])
        return pd.DataFrame(self.X, columns=cols)
