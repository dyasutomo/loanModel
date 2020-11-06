import logging
logger = logging.getLogger(__name__)

from sklearn.ensemble import VotingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier

#from lazypredict.Supervised import LazyClassifier
#from keras.models import Sequential, load_model
#from keras.layers import Dense, Dropout
#from keras.callbacks import EarlyStopping

import lightgbm as lgb
#from xgboost import XGBClassifier
from joblib import dump, load

class Models:
    ''' Input: training or test data
        Output: model or prediction
    '''

    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def build_network(self, n_features, n_layers=9, n_nodes=100, activ_func='relu', dropout_rate=0.2):
        ''' Building neural network model '''
        model = Sequential()
        model.add(Dense(n_nodes, activation=activ_func, use_bias=True, input_shape=(n_features,)))
        for i in range(n_layers-1):
            model.add(Dense(n_nodes, activation=activ_func, use_bias=True))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate, input_shape=(n_nodes,)))
        model.add(Dense(1, activation='sigmoid'))
        return model

    def neural_network(self):
        # Building neural network model
        n_features = np.shape(self.X)[1]
        self.model = self.build_network(n_features)
        # Compiling model
        early_stopping_monitor = EarlyStopping(patience=5)
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        # Fitting model
        history = self.model.fit(self.X, self.y,
                                 validation_split=0.2, epochs=50,
                                 shuffle=True, callbacks=[early_stopping_monitor])
        self.model.save('neuralnetwork.h5')
        return self.model

    def train_lgbm(self, model_file='lgbm.model'):
        self.model = lgb.LGBMClassifier()
        self.model.fit(self.X, self.y)
        dump(self.model, model_file)
        return self.model

    def train_xgb(self, model_file='xgb.model'):
        self.model = XGBClassifier()
        self.model.fit(self.X, self.y)
        dump(self.model, model_file)
        return self.model

    def train_bag(self, model_file='bagging.model'):
        self.model = BaggingClassifier()
        self.model.fit(self.X, self.y)
        dump(self.model, model_file)
        return self.model

    def ensemble_model(self):
        lgbm     = lgb.LGBMClassifier()
        xgb      = XGBClassifier()
        bag      = BaggingClassifier()
        logres   = LogisticRegression()
        ridge    = RidgeClassifier()
        self.ensemble = VotingClassifier(estimators=[('lgbm', lgbm),
                                                     ('xgb', xgb),
                                                     ('bag', bag),
                                                     ('logres', logres),
                                                     ('ridge', ridge)],
                                                     voting='hard')
        self.ensemble.fit(self.X, self.y)
        return self.ensemble

    def lazy_classifier(self, X_val, y_val):
        clf = LazyClassifier(verbose=1, ignore_warnings=True, custom_metric=None)
        self.models, self.predictions = clf.fit(self.X, X_val, self.y, y_val)
        return self.models, self.predictions

    def load(self, model_file):
        if model_file == 'neuralnetwork': self.model = load_model(model_file+'.h5')
        else: self.model = load(model_file+'.model')

    def evaluate(self, model_name=None):
        if model_name == 'neuralnetwork':
            return self.model.evaluate(self.X, self.y)
        return self.model.score(self.X, self.y)

    def predict(self):
        return self.model.predict(self.X)
