# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 11:58:30 2021

@author: jakub
"""



import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut,StratifiedKFold,KFold
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,recall_score,roc_auc_score,precision_score
import random
import seaborn as sns


def specifity(y,y_pred):
        tn, fp, fn, tp = confusion_matrix(y,y_pred).ravel()
        return (tn / (tn+fp))
def specifity_multi(cm):
    sp0 = (cm[1,1]+cm[1,2]+cm[2,1]+cm[2,2])/(np.sum(cm[1])+np.sum(cm[2]))
    sp1 = (cm[0,0]+cm[0,2]+cm[2,0]+cm[2,2])/(np.sum(cm[0])+np.sum(cm[2]))
    sp2 = (cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])/(np.sum(cm[0])+np.sum(cm[1]))
    return (sp1+sp2+sp0)/3

def plot_confusion_matrix(cm_tab, row_tab_lab,cm_title="Confusion Matrix"):
    # cm_tab lista z macierzami pomyłek, row_tab_lab : lista z nazwami wierszy
    #for i in cm: cm_tab.append(i.ravel()) 

    fig, ax = plt.subplots(figsize=(9.4, 6.8)) 
    ax.set_axis_off() 
    table = ax.table( 
    cellText =  cm_tab,
    rowLabels = row_tab_lab,  
    colLabels = ['tn','fp','fn','tp'], #tn, fp, fn, tp
    rowColours =["palegreen"] * 10,  
    colColours =["palegreen"] * 10, 
    cellLoc ='center',  
    loc ='upper left')
    ax.set_title(cm_title, fontweight ="bold")
    plt.show()

def plot_cm(conf_mat,classes,titles, cm_title="Macierz pomyłek"):
    # cm_tab lista z macierzami pomyłek, row_tab_lab : lista z nazwami wierszy
    #for i in cm: cm_tab.append(i.ravel()) 
    def plot_matrix(cm, classes, title,ax):
      g = sns.heatmap(cm, cmap="Blues", annot=True,annot_kws={"size": 15}, 
                      xticklabels=classes,yticklabels=classes, ax=ax,cbar=False)
      #g.set(title=title, xlabel="predicted label", ylabel="true label")
      g.set_xlabel("PREDICT", fontsize = 13)
      g.set_ylabel("REAL", fontsize = 13)
      g.set_title(title, fontsize = 15, weight='bold')
      g.set_yticklabels(classes, size = 13)
      g.set_xticklabels(classes, size = 13)
    sns.set_theme()
  
    fig,axs = plt.subplots(3,2,figsize = (9.4, 6.8))
    axs[2,1].axis('off')
    for cm,title,ax in zip(conf_mat,titles,axs.ravel()):
        plot_matrix(cm, classes, title,ax)
    
    for ax in axs.flat:
        ax.label_outer()
        
    fig.tight_layout()
    plt.show()


    
def plot_metrics(scores_tab,title,x_label,y_label,x_tick_name=[],dist=0.003):
    plt.subplots(1, 1, figsize = (12.4, 6.8))
    plt.style.use('ggplot')
    plt.clf()
    x_ticks=np.arange(1,len(x_tick_name)+1)
  
    cell_text=[]
    row_labels=[]
    colors=[]
    for k,v in scores_tab.items(): 
     
        move=random.uniform(1-dist/2,1+dist/2)
        y_ticks=[1 if el==1 or el*move>=1 else el*move for el in v ]
        base_plot=plt.plot(x_ticks,y_ticks,marker=(5,1),linestyle='-',label=k)
        cell_text.append(v)
        row_labels.append(k)
        colors.append(base_plot[-1].get_color())
        
        table = plt.table(cellText=cell_text,
                          rowLabels=row_labels,
                          rowColours=colors,
                          colLabels=x_tick_name,
                          cellLoc='center',
                          loc='bottom')
    
    plt.ylabel(y_label,fontsize=13)
    # plt.legend(bbox_to_anchor=(1.05, 1),
    #                          loc='upper left', borderaxespad=0)
    plt.title(title,fontsize=15)
    plt.xticks(x_ticks,[],rotation='vertical')
    plt.yticks(fontsize=13)
    table.set_fontsize(11)
    plt.show()
    


def calc_clf(clf,X_train,y_train,X_test,y_test,neighbours=0 ):
    #liczy klasyfikator dla zadanego zbioru , zwraca y_pred 
        if neighbours!=0: classifier = clf(neighbours)
        elif clf == SVC: classifier = clf(probability=True)
        elif clf == LogisticRegression :classifier = LogisticRegression(solver='newton-cg')
        else: classifier = clf()
       
        classifier.fit(X_train,y_train)
        y_pred= classifier.predict(X_test)
        y_pred_proba = classifier.predict_proba(X_test)
        return y_pred, y_pred_proba
    
def load_dataset():
    iris = datasets.load_iris()
    X=iris.data
    y=iris.target
    y=np.where(y==2,0,y)
    return X,y

def load_dataset_iris():
    iris = datasets.load_iris()
    X=iris.data
    y=iris.target
 
    return X,y

def load_dataset_wine():
    iris = datasets.load_wine()
    X=iris.data
    y=iris.target
#   y=np.where(y==2,0,y)
    return X,y

def load_dataset_zad1():
    data_set = np.array([1,5.3,2,
                         2.8,7.6,1,
                         4.2,9.3,2,
                         1.5,3.1,1,
                         9.8,7.5,2,
                         6.1,0.5,2,
                         4.7,8.9,2,
                         1.2,8,1,
                         8.2,3.3,1,
                         6.4,5.5,1]).reshape(-1,3)

    y= data_set[:,2].reshape(-1)
    y=np.where(y==2,0,y)
    X= data_set[:,:2]
    return X,y
"""
def confusion_matrix_scorer_tn(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    return {'tn': cm[0, 0], 'fp': cm[0, 1],
            'fn': cm[1, 0], 'tp': cm[1, 1]}  
 """


def cm_tn_scorer(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    return  cm[0, 0]

def cm_fp_scorer(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    return  cm[0,1]

def cm_fn_scorer(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    return  cm[1, 0]

def cm_tp_scorer(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    return  cm[1, 1]

def confusion_matrix_loo(y_test,y_pred):
   cm=[[0,0,0],[0,0,0],[0,0,0]]
   for y_t,y_p in zip(y_test, y_pred) : 
       if y_t==0 and y_p==0: cm=np.add(cm,[[1,0,0],[0,0,0],[0,0,0]])
       elif y_t==0 and y_p==1: cm=np.add(cm,[[0,1,0],[0,0,0],[0,0,0]])
       elif y_t==0 and y_p==2: cm=np.add(cm,[[0,0,1],[0,0,0],[0,0,0]])
       elif y_t==1 and y_p==0: cm=np.add(cm,[[0,0,0],[1,0,0],[0,0,0]])
       elif y_t==1 and y_p==1: cm=np.add(cm,[[0,0,0],[0,1,0],[0,0,0]])
       elif y_t==1 and y_p==2: cm=np.add(cm,[[0,0,0],[0,0,1],[0,0,0]])
       elif y_t==2 and y_p==0: cm=np.add(cm,[[0,0,0],[0,0,0],[1,0,0]])
       elif y_t==2 and y_p==1: cm=np.add(cm,[[0,0,0],[0,0,0],[0,1,0]])
       elif y_t==2 and y_p==2: cm=np.add(cm,[[0,0,0],[0,0,0],[0,0,1]])
   return cm 

# słownik metryk
metrics = {"Accuracy": accuracy_score, 
           "F1": f1_score,
           "Precision": precision_score,
           "Recall": recall_score,
           "Specificity" : specifity,
           "AUC": roc_auc_score,
           }

# słownik metryk
#metrics = {"Jakość": accuracy_score, 
 #          "F1": f1_score,
  #         "Precyzja": precision_score,
   #        "Czułość": recall_score,
    #       "Swoistosc" : specifity,
     #     }

# metrics_cv = {"Jakość": 'accuracy', 
#            "F1": 'f1',
#            "Precyzja": 'precision',
#            "Czułość": 'recall',
#            #"Specyficzność" : specifity,
#            "AUC": 'roc_auc',
#            "tn": cm_tn_scorer,
#            "tp": cm_tp_scorer,
#            "fn": cm_fn_scorer,
#            "fp": cm_fp_scorer,
#            }
 

# słownik z wynikami 
scores_tab_init= {k: [] for k in metrics.keys()}

#słownik klasyfikatorow

classifier = {
             "Bayes":{'func':GaussianNB,'param':0},
             "SVM":{'func':SVC,'param':0},
             "LogReg":{'func':LogisticRegression,'param':0},
             "1NN": {'func': KNeighborsClassifier,'param':1},
             "kNN": {'func': KNeighborsClassifier,'param':5},
            }
def metrics_calc(y_test,y_pred,y_pred_proba):
    


    metrics_clf={k: [] for k in metrics.keys()}
    conf_mat_clf=confusion_matrix_loo(y_test,y_pred) 
    metrics_clf['Accuracy'].append(accuracy_score(y_test, y_pred))
    metrics_clf['Precision'].append(precision_score(y_test, y_pred, average='macro',zero_division=1))
    metrics_clf['Recall'].append(recall_score(y_test, y_pred, average='macro'))
    metrics_clf['F1'].append(f1_score(y_test, y_pred, average='macro'))
    metrics_clf['AUC'].append(roc_auc_score(y_test,y_pred_proba,multi_class='ovr'))
    metrics_clf['Specificity'].append(specifity_multi(conf_mat_clf))
    return conf_mat_clf, metrics_clf


def kross_calc(split_func, X,y):
    
    metrics_loo={k: [] for k in metrics.keys()}
   
    conf_mat_loo=[]

    for k,clf in classifier.items():    # dla każdego klasfikatora

        conf_mat_clf=[[0,0,0],[0,0,0],[0,0,0]]
        metrics_clf={k: [] for k in metrics.keys()}
       
        for train_index, test_index in split_func.split(X,y): #dla każdego foldsa licz macierz i metryki
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]   
     
            y_pred,y_pred_proba=calc_clf(clf['func'],X_train,y_train,X_test,y_test,clf['param'])
            
            cm_temp=confusion_matrix_loo(y_test,y_pred)
            conf_mat_clf=np.add(conf_mat_clf,cm_temp) #sumuj macierz pomylek 
            metrics_clf['Accuracy'].append(accuracy_score(y_test, y_pred))
            metrics_clf['Precision'].append(precision_score(y_test, y_pred, average='macro',zero_division=1))
            metrics_clf['Recall'].append(recall_score(y_test, y_pred, average='macro'))
            metrics_clf['F1'].append(f1_score(y_test, y_pred, average='macro'))
            metrics_clf['AUC'].append(roc_auc_score(y_test,y_pred_proba,multi_class='ovr'))

            metrics_clf['Specificity'].append(specifity_multi(cm_temp))
           
       
        for k in metrics_clf.keys(): # usrednij metryki z n foldsów 
            metrics_loo[k].append(round(sum(metrics_clf[k])/len(metrics_clf[k]),2))
           
        conf_mat_loo.append(conf_mat_clf)
   


    return conf_mat_loo, metrics_loo

def loo_calc(split_func, X,y):
    
    metrics_loo={k: [] for k in metrics.keys()}
   
    conf_mat_loo=[]

    for k,clf in classifier.items():    # dla każdego klasfikatora

        conf_mat_clf=[[0,0,0],[0,0,0],[0,0,0]]
        metrics_clf={k: [] for k in metrics.keys()}
        y_pred_proba_tab=[]
        y_test_tab=[]
        y_pred_tab=[]
        for train_index, test_index in split_func.split(X,y): #dla każdego foldsa licz macierz i metryki
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]   
     
            y_pred,y_pred_proba=calc_clf(clf['func'],X_train,y_train,X_test,y_test,clf['param'])
            
            cm_temp=confusion_matrix_loo(y_test,y_pred)
            conf_mat_clf=np.add(conf_mat_clf,cm_temp) #sumuj macierz pomylek 
            y_pred_proba_tab.extend(y_pred_proba)
            y_test_tab.extend(y_test)
            y_pred_tab.extend(y_pred)
            
       
        conf_mat_loo.append(conf_mat_clf)
        metrics_loo['Accuracy'].append(round(accuracy_score(y_test_tab, y_pred_tab),2))
        metrics_loo['Precision'].append(round(precision_score(y_test_tab, y_pred_tab, average='macro',zero_division=1),2))
        metrics_loo['Recall'].append(round(recall_score(y_test_tab, y_pred_tab, average='macro'),2))
        metrics_loo['F1'].append(round(f1_score(y_test_tab, y_pred_tab, average='macro'),2))
        metrics_loo['AUC'].append(round(roc_auc_score(y_test_tab,y_pred_proba_tab,multi_class='ovr'),2))
        metrics_loo['Specificity'].append(round(specifity_multi(conf_mat_clf),2))
           


    return conf_mat_loo, metrics_loo

def zad1():
    data_set = np.array([1,5.3,2,
                     2.8,7.6,1,
                     4.2,9.3,2,
                     1.5,3.1,1,
                     9.8,7.5,2,
                     6.1,0.5,2,
                     4.7,8.9,2,
                     1.2,8,1,
                     8.2,3.3,1,
                     6.4,5.5,1]).reshape(-1,3)

    y= data_set[:,2].reshape(-1)
    y=np.where(y==2,0,y)
    X= data_set[:,:2]

    X_train=X_test=X
    y_train=y_test=y

    neighbors=np.arange(1,len(X_train))
    result_metrics = {k: [] for k in metrics.keys()}
    result_conf_mat=[]
    x_label_names = [s for s in classifier.keys()]
    clf=classifier["kNN"]

    for i,n in enumerate(neighbors) :
   
        cm_temp=[]
        metrics_temp={k: [] for k in metrics.keys()}
        y_pred,y_pred_proba=calc_clf(clf['func'],X_train,y_train,X_test,y_test,n)
        result_conf_mat.append(confusion_matrix(y_test,y_pred).ravel())
        for k,met in metrics.items():
            result_metrics[k].append(round(met(y_test,y_pred),2))

    
    plot_metrics(result_metrics,"Metrics for KNN <1,9>","Classifers","Metrics",['knn='+str(x) for i,x in enumerate(neighbors)],dist=0.04)
    plot_confusion_matrix(result_conf_mat,['knn='+str(x) for i,x in enumerate(neighbors)])

def zad2_redyst(X,y):
    X_train=X_test=X
    y_train=y_test = y
    x_label_names = [s for s in classifier.keys()]
    metrics_redyst={k: [] for k in metrics.keys()}
    conf_mat_redyst=[]


    #wyliczanie metryk
    for i,clf in classifier.items() :
        cm_temp=[]
        metrics_temp={}
        y_pred,y_pred_proba=calc_clf(clf['func'],X_train,y_train,X_test,y_test,clf['param'])
        cm_temp,metrics_temp=metrics_calc(y_test,y_pred,y_pred_proba)
        conf_mat_redyst.append(cm_temp)
       
        for k in metrics_redyst.keys(): 
            metrics_redyst[k].append(round(metrics_temp[k][0],2))
       
    plot_metrics(metrics_redyst," Redistribution ","Classifiers ","Metrics",x_label_names,dist=0.02)
    plot_cm(conf_mat_redyst,[' class 0',' class 1',' class 2'],[x for x in classifier.keys()])

def zad2_podzial(X,y):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=21,stratify=y)

    x_label_names = [s for s in classifier.keys()]
    metrics_podz={k: [] for k in metrics.keys()}
    conf_mat_podz=[]


    #wyliczanie metryk
    for i,clf in classifier.items() :
        cm_temp=[]
        metrics_temp={}
        y_pred,y_pred_proba=calc_clf(clf['func'],X_train,y_train,X_test,y_test,clf['param'])
        cm_temp,metrics_temp=metrics_calc(y_test,y_pred,y_pred_proba)
        conf_mat_podz.append(cm_temp)
       
        for k in metrics_podz.keys(): 
            metrics_podz[k].append(round(metrics_temp[k][0],2))
       
   
    plot_metrics(metrics_podz,"Split 70/30","Klasyfikatory","Metrics",x_label_names,dist=0.02) 
    plot_cm(conf_mat_podz,[' class 0',' class 1',' class 2'],[x for x in classifier.keys()])

def zad2_kross(X,y):
    cm_kross,mr_loo = kross_calc(StratifiedKFold(n_splits=5),X,y)
    plot_metrics(mr_loo,"Crossvalidation, k = 5 ","Klasyfikatory","Metrics",[s for s in classifier.keys()],dist=0.02)
    plot_cm(cm_kross,[' class 0',' class 1',' class 2'],[x for x in classifier.keys()])

def zad2_loo(X,y):
    cm_kross,mr_loo = loo_calc(LeaveOneOut(),X,y)
    plot_metrics(mr_loo,"LeaveOneOut","Klasyfikatory","Metrics",[x for x in classifier.keys()],dist=0.02)
    plot_cm(cm_kross,[' class 0',' class 1',' class 2'],[x for x in classifier.keys()])

