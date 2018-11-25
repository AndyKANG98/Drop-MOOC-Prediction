"""
logistic regression
Neural network, decision trees
Random forests
Naive Bayesian
"""
import pandas as pd
import time
import numpy as np
from preprocess import Preprocessor
# Scikit Learn Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier

# Path
log_train_path = "data/train/log_train.csv"
enroll_train_path = "data/train/enrollment_train.csv"
truth_train_path = "data/train/truth_train.csv"
log_test_path = "data/test/log_test.csv"
enroll_test_path = "data/test/enrollment_test.csv"
truth_test_path = "data/test/truth_test.csv"

def random_forest(X_train, y_train, X_test, y_test, N_estimators=500):

    clf = RandomForestClassifier(n_estimators=500)
    clf = clf.fit(X_train, y_train)
    expected = y_test
    predicted = clf.predict(X_test)

    print("Classification report for classifier %s:\n%s\n"
          % (clf, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
    print 'Score: '
    print clf.score(X_test,y_test)

def SVM(X_train, y_train, X_test, y_test, Gamma='scale'):

    clf = svm.SVC(gamma='scale') 
    clf = clf.fit(X_train, y_train)
    expected = y_test
    predicted = clf.predict(X_test)

    print("Classification report for classifier %s:\n%s\n"
          % (clf, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
    print 'Score: '
    print clf.score(X_test,y_test)
    print 'Cross validation score: '
    print cross_val_score(clf, X_train, y_train, cv=3)
    
def MLPClassifier(X_train, y_train, X_test, y_test):
    
    clf = MLPClassifier()
    clf = clf.fit(X_train, y_train)
    expected = y_test
    predicted = clf.predict(X_test)

    print("Classification report for classifier %s:\n%s\n"
          % (clf, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
    print 'Score: '
    print clf.score(X_test,y_test)
    print 'Cross validation score: '
    print cross_val_score(clf, X_train, y_train, cv=3)
    
    
def main():
    
    # Build preprocessor
    p_train = Preprocessor(log_train_path, enroll_train_path, truth_train_path)
    p_test = Preprocessor(log_test_path, enroll_test_path, truth_test_path)
    
    # Get value
    X_train, y_train = p_train.get_values_all()
    X_test, y_test = p_test.get_values_all()
    
    # Random forest: N_estimators
    random_forest(X_train, y_train, X_test, y_test, N_estimators=500)
    
    # SVM: gamma
    SVM(X_train, y_train, X_test, y_test, Gamma='scale')
    
    # Neural network
    neural_network(X_train, y_train, X_test, y_test)
    


if __name__== "__main__":
  main()