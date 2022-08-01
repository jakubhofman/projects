# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 08:56:13 2022

@author: jakub
"""

import pandas as pd

from sklearn.feature_selection import  RFECV,SelectFromModel,f_classif,SelectKBest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import time

#wczytawanie dane DLBCL
def load_data(training_file, testing_file):
    DX_test=pd.read_csv(testing_file)
    DX_train=pd.read_csv(training_file)
    
    Dy_train=DX_train['class']
    DX_train.drop(labels=['class'], axis=1, inplace = True)
    
    Dy_test=DX_test['class']
    DX_test.drop(labels=['class'], axis=1, inplace = True)   
    
    return DX_train,Dy_train,DX_test,Dy_test


def test_jakosci(X_train_T,y_train,X_test_T,y_test):
    # testuje jakosc wybranego zestawu cech zwraca miarę accuracy dla każdego z klasyfikatorów 
    clf=GaussianNB()
    clf.fit(X_train_T,y_train)
    y_pred= clf.predict(X_test_T)
    q_NBC=accuracy_score(y_test, y_pred)
    
    clf=KNeighborsClassifier(5)
    clf.fit(X_train_T,y_train)
    y_pred= clf.predict(X_test_T)
    q_5NN=accuracy_score(y_test, y_pred)
    
    clf=DecisionTreeClassifier(random_state=137)
    clf.fit(X_train_T,y_train)
    y_pred= clf.predict(X_test_T)
    q_J48=accuracy_score(y_test, y_pred)
    
    return (1-q_NBC+1-q_5NN+1-q_J48)/3

#filtrowanie za pomocą SelectPercentile i funkcji scoringowej f_classif 
def sel_f_classif(X_train,y_train,X_test,y_test,procent):
    start = time.time()
    X_sp = SelectKBest(f_classif, k=procent)
    X_train_T= X_sp.fit_transform(X_train, y_train)
    X_test_T= X_test.iloc[:,X_sp.get_support()].values
    end = time.time()
    return test_jakosci(X_train_T,y_train,X_test_T,y_test),sum(X_sp.get_support()),(end-start)

#selekcja RFE  kalsyfikator SVC
def sel_RFE(X_train,y_train,X_test,y_test,feat):
    start = time.time()
    svc = SVC(kernel="linear")
    rfecv = RFECV(
    estimator=svc,
    step=10,
    cv=2,
    scoring="accuracy",
    min_features_to_select=feat,
    )
    X_train_T=rfecv.fit_transform(X_train, y_train)
    X_test_T= X_test.iloc[:,rfecv.get_support()]
    end = time.time()
    return test_jakosci(X_train_T,y_train,X_test_T,y_test),sum(rfecv.get_support()),(end-start)

def sel_Tree(X_train,y_train,X_test,y_test,param):
    start = time.time()
    sel_ran = SelectFromModel(estimator=DecisionTreeClassifier(max_features=param, random_state=137))
    X_train_T=sel_ran.fit_transform(X_train, y_train)
    X_test_T= X_test.iloc[:,sel_ran.get_support()].values
    end = time.time()
    return test_jakosci(X_train_T,y_train,X_test_T,y_test),sum(sel_ran.get_support()),(end-start)

def sel_RandomForest(X_train,y_train,X_test,y_test,param):
    start = time.time()
    sel_ran = SelectFromModel(estimator=RandomForestClassifier(max_features=param, random_state=137))
    X_train_T=sel_ran.fit_transform(X_train, y_train)
    X_test_T= X_test.iloc[:,sel_ran.get_support()].values
    end = time.time()
    return test_jakosci(X_train_T,y_train,X_test_T,y_test),sum(sel_ran.get_support()),(end-start)


metody = {
            "Anova":sel_f_classif,
            "RFE":sel_RFE,
            "Tree":sel_Tree,
            "Random":sel_RandomForest
            }

def select_feature(X_train,y_train,X_test,y_test,num_atr):
    wynik=[]
    for i,func in metody.items():
      for proc in num_atr:
          feat = round(proc*len(X_train.columns)/100)
          a,b,t = func(X_train,y_train,X_test,y_test,feat)
          wynik.append([i,round(a,3),b,round(t,3),proc])
    wynik.append(['Bez_selekcji',test_jakosci(X_train,y_train,X_test,y_test),0,0,0])
    
    return pd.DataFrame(wynik,columns =['Algorytm', 'Błąd', 'L.cech','Czas','% cech']) 

