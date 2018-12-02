import pandas as pd
import time
import numpy as np
from preprocess import Preprocessor
# Scikit Learn Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn import metrics
# XGBoost Library
from xgboost import XGBClassifier

def random_forest(X_train, y_train, X_test, y_test):

    clf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=0, min_samples_split=2, n_jobs=-1)
    clf = clf.fit(X_train, y_train)
    expected = y_test
    predicted = clf.predict(X_test)

    print("Classification report for classifier %s:\n%s\n"
          % (clf, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
    print ("Testing Score:")
    print (clf.score(X_test,y_test))
    print ('')
    
def logistic_regression(X_train, y_train, X_test, y_test):
    
    clf = LogisticRegression(tol=1e-3, C=1.5, random_state=0)
    clf = clf.fit(X_train, y_train)
    expected = y_test
    predicted = clf.predict(X_test)

    print("Classification report for classifier %s:\n%s\n"
          % (clf, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
    print ("Testing Score:")
    print (clf.score(X_test,y_test))
    print ('')
    
def XGboost(X_train, y_train, X_test, y_test):

    clf = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, min_child_weight=2, 
                        n_jobs=-1, max_delta_step=1, objective='binary:logistic', gamma=3 ,subsample=1)
    clf = clf.fit(X_train, y_train)
    expected = y_test
    predicted = clf.predict(X_test)

    print("Classification report for classifier %s:\n%s\n"
          % (clf, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
    print ("Testing Score:")
    print (clf.score(X_test,y_test))
    print ('')
    
def voting(X_train, y_train, X_test, y_test):
    
    clf1 = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=0, min_samples_split=3, n_jobs=-1)
    clf2 = LogisticRegression(tol=1e-3, C=1.5, random_state=0)
    clf3 = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, min_child_weight=2, 
                        n_jobs=-1, max_delta_step=1, objective='binary:logistic', gamma=3 ,subsample=1)
    
    clf = VotingClassifier(estimators=[('rf', clf1), ('lr', clf2), ('xgb', clf3)], voting='hard')
    clf = clf.fit(X_train, y_train)
    expected = y_test
    predicted = clf.predict(X_test)

    print("Classification report for classifier %s:\n%s\n"
          % (clf, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
    print ("Testing Score:")
    print (clf.score(X_test,y_test))
    
def main():
    
    # Build preprocessor
    p_train = Preprocessor('train')
    p_test = Preprocessor('test')
    
    # Get value
    X_train, y_train = p_train.get_values_all()
    X_test, y_test = p_test.get_values_all()
    
    print ('==========Classification==========')
    # Random forest
    random_forest(X_train, y_train, X_test, y_test)
    
    # Logistic regression
    logistic_regression(X_train, y_train, X_test, y_test)
    
    # XGboost
    XGboost(X_train, y_train, X_test, y_test)
    
    print ('==========Voting==========')
    # Voting
    voting(X_train, y_train, X_test, y_test)

if __name__== "__main__":
    main()