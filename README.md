# loanModel
Predicting the outcome of loan based on LendingClub data  
The deployment of this model is in https://github.com/dyasutomo/deploy-loanModel

You can execute it by python main.py --argsname  
where argsname could be:  
    inputfile : filename for training or predicting in csv format  
    outputfile: output file that record the predicted outcome  
    select_columns : columns to be selected for modeling  
    categorical_columns : list of categorical columns  
    ordered_columns : list of ordered columns  
    numerical_columns : list of numerical columns  
    target_column : defaulted to 'loan_status'  
    testsize : percentage of data for testing  
    impute : NaN imputation ('median', 'mean', 'mode')  
    model_name : model to be used (lgbm, xgb, bagging, neuralnetwork)  
    mode : 'train' for training, 'predict' for predicting  
