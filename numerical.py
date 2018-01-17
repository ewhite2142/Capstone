import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.utils import resample
import time
from alchemy_conn import alchemy_engine

#inherently multiclass:
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

#multiclass as One-vs-One:
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier #set multi_class="one_vs_one"

#multiclass as One-Vs-All:
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron

from catboost import CatBoostClassifier #non-sklearn classifer with good reputation


def print_numsummits(df):
    '''
    INPUT: DataFrame containing summits table
    OUTPUT: None--prints number of mounts, mountains, and peaks
    '''
    num_mountains = len(df[(df['type_str'] == 'mountain')]['summit_id'].unique())
    num_mounts = len(df[(df['type_str'] == 'mount')]['summit_id'].unique())
    num_peaks = len(df[(df['type_str'] == 'peak')]['summit_id'].unique())
    print("#mountains={}. #mounts={}, #peaks={}, total={}".format(num_mountains, num_mounts, num_peaks, num_mountains+num_mounts+num_peaks))


def read_prepare_data():
    '''
    INPUT: none
    OUTPUT: DataFrames X_train, X_test, y_train, y_test, df

    Read data from summits table in summitsdb into df.
    Puts features 'elevation','isolation', and 'prominence' into X.
    Puts type (0=Mount, 1=Mountain, 2=Peak) into y
    Standardizes X
    Splits X, y into X_train, X_test, y_train, y_test'''

    #use imported alchemy_conn program to generate sqlalchemy connection to summitsdb
    engine = alchemy_engine()

    #load summits table into pandas df
    # df = pd.read_sql_query('''SELECT * FROM summits ORDER BY summit_id;''', con=engine)
    df = pd.read_csv('~/dsi/Capstone/summits.csv')
    X = df[['elevation','isolation', 'prominence']]
    y = df['type']

    #standardize features data
    scalar = StandardScaler(copy=True, with_mean=True, with_std=True)
    scalar.fit(X)
    X = scalar.transform(X)

    #split into train, test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

    return X_train, X_test, y_train, y_test, df



def pick_best_classifier(X_train, X_test, y_train, y_test):
    '''
    INPUT: four DataFrames
    OUTPUT: fitted model (the one with the best f1 score), DataFrame showing score results for each classifier

    Loops through multiple classifiers, performing GridSearch with cross validation on training data, and prints train and test results (f1, precision, recall, and accuracy scores) for each. The classifier (fitted on the training data) with the best f1 score is returned.
    '''

    # these are the classifiers we are testing:
    names = ['LogisticRegression', 'LogisticRegression--liblinear/ovr', 'DecisionTreeClassifier', 'GaussianNB',
             'LinearDiscriminantAnalysis', 'QuadraticDiscriminantAnalysis',
             'LinearSVC', 'MLPClassifier', 'RandomForestClassifier', 'SVC',
             'GradientBoostingClassifier', 'SGDClassifier', 'Perceptron']

    classifiers = [
        LogisticRegression(random_state=1, max_iter=1000),
        LogisticRegression(random_state=1, max_iter=1000, multi_class='ovr', solver='liblinear'),
        DecisionTreeClassifier(random_state=1),
        GaussianNB(),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis(),
        LinearSVC(random_state=1),
        MLPClassifier(),
        RandomForestClassifier(),
        SVC(),
        GradientBoostingClassifier(),
        SGDClassifier(),
        Perceptron()
    ]

    #params for GridSearchCV
    params=[
        {'multi_class': ['ovr', 'multinomial'],
             'solver': ['lbfgs', 'sag', 'saga', 'newton-cg'],
             'class_weight': [None, 'balanced']}, #LogisticRegression

        {'class_weight': [None, 'balanced']}, #LogisticRegression: liblinear/ovr
        {}, #DecisionTreeClassifier
        {}, #GaussianNB
        {'solver': ['svd', 'lsqr', 'eigen']}, #LinearDiscriminantAnalysis
        {}, #QuadraticDiscriminantAnalysis
        {'multi_class': ['ovr', 'crammer_singer'], 'class_weight': [None, 'balanced']}, #LinearSVC
        {}, #MLPClassifier
    #     {}, #RadiusNeighborsClassifier
        {}, #RandomForestClassifier
        {}, #SVC
        {'loss': ['deviance'], 'n_estimators': [50,100,200], 'max_depth': [2,3,5,7], 'criterion': ['friedman_mse'], 'max_features': [None, 'auto', 'sqrt', 'log2']}, #GradientBoostingClassifier
        {}, #SGDClassifier
        {}  #Perceptron
            ]

    #starting default values
    best_test_score = -999.9
    worst_test_score = 999.9
    best_estimator = ''
    longest_time = -1
    scores = dict()

    #loop through each classifier
    # for i in range(1): #for testing
    print()
    i=0
    for i in range(len(params)):
        print("=============================== {}. {} ==================================".format(i+1, names[i]))
        score_type = 'accuracy'
        start_time = time.time()

        #GridSearchCV below uses 3 fold cross validation, and searches through parameters in param_grid above for each classifier
        gs = GridSearchCV(estimator=classifiers[i], param_grid=params[i], cv=8, n_jobs=-1, scoring=score_type)
        gs = gs.fit(X_train, y_train)
        seconds = time.time() - start_time
        if seconds > longest_time:
            longest_time = seconds
            longest_time_estimator = names[i]

        #predict results (y_pred) with best_estimator (one with best parameters from GridSearchCV)
        model = gs.best_estimator_
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        #calculate and print scores
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        accuracy = accuracy_score(y_test, y_pred)
        test_score = f1 #f1 used to rank classifiers

        #print results
        print("TEST {} score: {}".format(score_type, test_score))
        print("TRAIN best {} score={}".format(score_type, gs.best_score_))
        if params[i] == {}:
            cur_params = "params: default used"
        else:
            cur_params = "best params: {}".format(gs.best_params_)
        print(cur_params)
        print("#seconds for GridSearchCV for this classifier={}\n".format(seconds))
        print("TEST scores:\nf1: {}\nprecision: {}\nrecall: {}\naccuracy: {}\n".format(f1, precision, recall, accuracy))
        print("\nconfusion matrix:\n    TN    FP\n    FN    TP\n{}\n".format(confusion_matrix(y_test, y_pred)))

        #store scores for printing summary later
        scores[names[i]] = (f1, precision, recall, accuracy)

        #record best and worst results from all classifiers
        if test_score > best_test_score:
            second_best_estimator = best_estimator
            second_best_score = best_test_score
            best_test_score = test_score
            best_estimator = names[i]
            best_params = cur_params
            best_estimator_seconds = seconds
            best_model = model
        if test_score < worst_test_score:
            worst_test_score = test_score
            worst_estimator = names[i]
            worst_params = cur_params
            worst_estimator_seconds = seconds

    #after running each classifer, print summary results
    print()
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("=============================== SUMMARY ==================================")
    print("best_estimator={}\nparams={}\nf1 test score={}\n#seconds={}".format(best_estimator, best_params, best_test_score, best_estimator_seconds))
    print("\nsecond_best_estimator: {}, f1 score: {}".format(second_best_estimator, second_best_score))
    print("\nworst_estimator={}, #seconds={}, params={}, f1 test score={}".format(worst_estimator, worst_estimator_seconds, worst_params, worst_test_score))
    print("\nestimator that took most time: {}, seconds: {}".format(longest_time_estimator, longest_time))

    #print scores summary: rows are each classifer, columns f1, precision, recall, accuracy
    scores = pd.DataFrame(scores).T
    scores.columns = ['f1', 'precision', 'recall', 'accuracy']
    scores = scores.sort_values('f1', ascending=False)
    print("\nSummary of results:\n{}".format(scores))

    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
    return best_model, scores #returns classifer with best f1 score and best parameters from GridSearchCV

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, df = read_prepare_data()
    print_numsummits(df)
    best_model, scores = pick_best_classifier(X_train, X_test, y_train, y_test)
