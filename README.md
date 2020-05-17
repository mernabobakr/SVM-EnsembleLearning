  # Assignment 3

## Building machine learning individual models 


### Baye's classifier
  * No of samples in Train data  :60,000
 * No of samples in Test data  :10,000
* Best accuracy reached: 55.58 %   
 

```python
def Bayesian_classifier(X_train,y_train,X_test,y_test):   
    
    clf = GaussianNB()
    clf.fit(X_train,y_train)
    GaussianNB(priors=None)
    y_pred = clf.predict(X_test)
    print(accuracy_score(y_test, y_pred))

```

### Support vector machine :
  * No of samples in Train data  :60,000
 * No of samples in Test data  :10,000
* Best accuracy reached: 98 %   
 

```python

def Support_Vector_classifier(X_train,y_train,X_test,y_test):     
    svm = SVC(kernel='poly',degree=2)
    svm.fit(X_train,y_train)
    pred = svm.predict(X_test)
    Support_Vector_classifier_accuracy=accuracy_score(pred,y_test)
    print(Support_Vector_classifier_accuracy)

```

### Desicion tree :
  * No of samples in Train data  :60,000
 * No of samples in Test data  :10,000
* Best accuracy reached : 88 %   
 

```python
def decsion_tree_classifier(X_train,y_train,X_test,y_test): 
    decision_tree_classifier = DecisionTreeClassifier()
    decision_tree_classifier.fit(X_train,y_train)
    pred = decision_tree_classifier.predict(X_test)
    decision_tree_classifier_accuracy=accuracy_score(pred,y_test)
    print(decision_tree_classifier_accuracy)
```
 
 ## Ensemble learning

### Voting classifier using desicion tree ,svm ,baye's classifier :
 * No of samples in Train data  :60,000
 * No of samples in Test data  :10,000
* Best accuracy reached : 93.78 %   


```python
def voting_Algorithm(X_train,y_train,X_test,y_test): 
    svm = SVC(kernel='poly',degree=2,probability=True)
    decision_tree = DecisionTreeClassifier()
    NB = GaussianNB()
    
    voting_classifier = VotingClassifier(estimators=[('svc', svm), ('dt', decision_tree), ('gnb', NB)], voting='soft')
    
    voting_classifier = voting_classifier.fit(X_train,y_train)
    y_pred=voting_classifier.predict(X_test)
    print(accuracy_score(y_test, y_pred))

```
   
### Bagging and pasting using Voting classifier  :

###  Bagging and pasting :
* Instead of running various models on a single dataset, you can use a single model over various random subsets of the dataset.
* I am using voting classifier so I could improve the accuracy.
* The No of samples used is 10 000 only as it was taking so much time.


| Bagging                             | Pasting         
| ----------------------------------- |:--------------------:
| Random sampling with replacement    | Random sampling without replacement                           
| Bootstrap parameter = True     | Bootstrap parameter = False  




```python
def Enesmble_Bagging_pasting(classifier,replacement):
    bg = BaggingClassifier(base_estimator=classifier, n_estimators=10, random_state=314,bootstrap=replacement)
    bg.fit(X_train,y_train)
    pred=bg.predict(X_test)
    acc=accuracy_score(pred,y_test)
    print(acc)

```
#### Pasting using Voting classifier  :
  * No of samples in Train data  :10,000
 * No of samples in Test data  :200
* Best accuracy reached : 93 %   
* bootstrap parameter = True 


#### Bagging using Voting classifier  :
  * No of samples in Train data  :10,000
 * No of samples in Test data  :200
* Best accuracy reached : 92.6  %   
*  bootstrap parameter = False 
