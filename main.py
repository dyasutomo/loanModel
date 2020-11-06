#!/usr/bin/env python

# Importing necessary modules
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging

from preprocess import *
from premodel import *
from models import *

def training(inputfile, target_column, select_columns, categorical_columns,
             ordered_columns, numerical_columns, model_name, impute, test_size):

    """ For training model and save it.
    """

    # Importing data
    df = pd.read_csv(inputfile)
    logging.info('Succesfully read %s' % inputfile)
    logging.info('Dimension of raw data is %i x %i' % (df.shape[0], df.shape[1]))

    # Reading columns
    select_columns = np.loadtxt(select_columns, dtype='str').tolist()
    categorical_columns = np.loadtxt(categorical_columns, dtype='str').tolist()
    ordered_columns = np.loadtxt(ordered_columns, dtype='str').tolist()
    numerical_columns = np.loadtxt(numerical_columns, dtype='str').tolist()

    # Cleaning data
    logging.info('--> Preprocessing Data')
    df = Preprocess(df)
    df.reduce_data(select_columns, target_column, classification=['Fully Paid', 'Charged Off'])
    df.make_catcols(categorical_columns)
    df.make_ordcols(ordered_columns)
    df.str_to_num(colname='emp_length', suffix=' years', replace={'< 1':'0', '10+':'10'})
    df.calc_time_interval(time1='earliest_cr_line', time2='issue_d', time_diff='time_interval')
    df = df.impute_nan(impute='median')

    # Prepare data before modeling
    df = Premodel(df, 'loan_status')
    X, y = df.upsampling()
    del X; del y # to save memory
    X_train, X_test, y_train, y_test = df.splitting(0.2)
    X_train = df.scaling_fit_transform(numerical_columns)
    del df # to save memory

    # Train model
    model = Models(X_train, y_train)
    if model_name == 'lgbm':
        model.train_lgbm()
        logging.info('Accuracy of LGBM is %.3f' % model.evaluate())
    elif model_name == 'xgb':
        model.train_xgb()
        logging.info('Accuracy of XGB is %.3f' % model.evaluate())
    elif model_name == 'bagging':
        model.train_bag()
        logging.info('Accuracy of Bagging is %.3f' % model.evaluate())
    elif model_name == 'neuralnetwork':
        model.neural_network()
        logging.info('Accuracy of Neural Network is %.3f' % model.evaluate(model_name))
    else:
        raise Exception('Please define your --model input: lgbm, xgb, bagging, or neuralnetwork')

def predicting(inputfile, outputfile, numerical_columns, model_name):
    """ For predicting model """
    df_test = pd.read_csv(inputfile)
    numerical_columns = np.loadtxt(numerical_columns, dtype='str').tolist()
    user_input = Premodel(df_test).scaling_transform(numerical_columns)
    model = Models(user_input)
    model.load(model_name)
    y_pred = model.predict()
    pd.Series(y_pred.reshape(-1)).to_csv(outputfile, index=False)
    return y_pred

# Main code
if __name__ == "__main__":

    # Defining inputs
    parser = argparse.ArgumentParser(description='Predict loan')
    parser.add_argument('--inputfile', type=str, default='../data/accepted_2007_to_2018Q4.csv')
    parser.add_argument('--outputfile', type=str, default='predictions.csv')
    parser.add_argument('--select_columns', type=str, default='select_columns.txt')
    parser.add_argument('--categorical_columns', type=str, default='categorical_columns.txt')
    parser.add_argument('--ordered_columns', type=str, default='ordered_columns.txt')
    parser.add_argument('--numerical_columns', type=str, default='numerical_columns.txt')
    parser.add_argument('--target_column', type=str, default='loan_status')
    parser.add_argument('--testsize', type=float, default=0.2)
    parser.add_argument('--impute', type=str, default='median')
    parser.add_argument('--model_name', type=str, default='lgbm')
    parser.add_argument('--mode', type=str, default='train')
    args = parser.parse_args()

    # Initialize logging
    logging.basicConfig(level=logging.INFO, filename='logfile.txt')
    #logging.disable('logging.DEBUG')
    logging.info('------ Running app ------')

    # Executing task
    if args.mode == 'train':
        training(inputfile=args.inputfile,
                 target_column=args.target_column,
                 select_columns=args.select_columns,
                 categorical_columns=args.categorical_columns,
                 ordered_columns=args.ordered_columns,
                 numerical_columns=args.numerical_columns,
                 model_name=args.model_name,
                 impute=args.impute,
                 test_size=args.testsize)

    elif args.mode == 'predict':
        predicting(inputfile=args.inputfile,
                   outputfile=args.outputfile,
                   numerical_columns=args.numerical_columns,
                   model_name=args.model_name)

    else:
        raise Exception("Choose train or predict as an input to mode")
