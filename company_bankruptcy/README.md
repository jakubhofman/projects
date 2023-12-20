__Company bankruptcy__

The project concerned the creation of model that predicts the bankruptcy of a company 
basing on true, financial data and F1 as a metric. 


**Result**: Final model beats the best results of internal (data workshop) Kaggle competition.


**Final model** is  ensmeble model composed of 'XGBClassifier', 'CatBoostClassifier' and 'GaussianNB'. Basics models were
trained using proba threshold set on 0.1 with selected features for each model. Basic models predictions were merged 
using voting : three 'ones' neccessary to set class 1 in final prediction. Projects experiments involved
set of different basic models: 'LGBMClassifier', 'XGBClassifier', 'CatBoostClassifier',
'GaussianNB', 'RandomForestClassifier', proba thresholds, featrue selections and ensemble methods. 

***Key factor to achieve best results :***

**proba_threshold** - expermineted with different values. Eventually 0.1 raised the metric by 12 points.

**Feature selection** : for each basic model specific set of features was selected. Method of selection involved 
calculating different metrics based on features weights for each class and  each features :  min, max, mean, 
sum_positive, sum_negative, count_positive, count_negative, mean_positive where for example mean_positive is average of 
feature weights with positive value for specific class. Based on these  metrics different feature selection method  was 
created for exmaple mean_dif_1 - this method choose features with the biggest difference between mean_positive and 
mean_negative for class 1. Each method was tested with different set of features and eventualy best set of features of all
method was chosen as final one for the basic model.

**Ensemble learning**: two different approaches were used. Meta model (logistic regression) to merged the prediction
of basic model and voting. Different set of models were tested. Final model described above. 

**Hiperparemeters** : tested but eventually not included in final model due to poor results of optimization.



