import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.svm import SVC




#decision tree
def decsion_tree_classifier():
    decision_tree_classifier = DecisionTreeClassifier()
    decision_tree_classifier.fit(X_train,y_train)
    pred = decision_tree_classifier.predict(X_test)
    decision_tree_classifier_accuracy=accuracy_score(pred,y_test)
    print(decision_tree_classifier_accuracy)
    

#Support Vector 
def Support_Vector_classifier():   
    svm = SVC(kernel='poly',degree=2)
    svm.fit(X_train,y_train)
    pred = svm.predict(X_test)
    Support_Vector_classifier_accuracy=accuracy_score(pred,y_test)
    print(Support_Vector_classifier_accuracy)
    


#bayesian-rule
def Bayesian_classifier():   
    
    
    clf = GaussianNB()
    clf.fit(X_train,y_train)
    GaussianNB(priors=None)
    y_pred = clf.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    


def voting_Algorithm():
    svm = SVC(kernel='poly',degree=2,probability=True)
    decision_tree = DecisionTreeClassifier()
    NB = GaussianNB()
    
    voting_classifier = VotingClassifier(estimators=[('svc', svm), ('dt', decision_tree), ('gnb', NB)], voting='soft')
    return voting_classifier
###    voting_classifier = voting_classifier.fit(X_train,y_train)
###    y_pred=voting_classifier.predict(X_test)
###    print(accuracy_score(y_test, y_pred))
    
def Enesmble_Bagging_pasting(classifier,replacement):
    bg = BaggingClassifier(base_estimator=classifier, n_estimators=10, random_state=314,bootstrap=replacement)
    bg.fit(X_train,y_train)
    pred=bg.predict(X_test)
    acc=accuracy_score(pred,y_test)
    print(acc)

train = pd.read_csv('mnist_train.csv')
train=train.head(10000)
test = pd.read_csv('mnist_test.csv')
test=test.head(200)

#Preparing the Training and Testing Data
X_train = train.drop(['label'], axis=1)
y_train = train['label']

X_test = test.drop(['label'], axis=1)
y_test = test['label']



#using pasting and voting the accuracy is 93.4%
#Enesmble_Bagging_pasting(voting_Algorithm(),False)

#using Bagging and voting the accuracy is 92.3%
#Enesmble_Bagging_pasting(voting_Algorithm(),True)


#using voting algorithm the accuracy is 93.78
#voting_Algorithm()

#decsion_tree_classifier()
# decision tree accuracy is accuracy is 88%

#Support_Vector_classifier()
#support vector machine accuracy is 98%

#Bayesian_classifier()
#Bayesian_classifier accuracy is 55.58 %