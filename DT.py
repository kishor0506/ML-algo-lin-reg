# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 19:15:02 2020

@author: ADMIN
"""

## 
import pandas as pd
import numpy as np
data = pd.read_csv("train.csv")


data.isna().sum()
data["Age"].fillna(data.Age.mean(skipna=True), inplace = True)

del data["Cabin"]
data.drop(columns =["Ticket","Name"], axis = 1, inplace = True )


data.dtypes
data.Embarked.value_counts()
data["Embarked"].fillna('S', inplace = True)

#### Label Encoding 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(data['Sex'].astype(str))
data['Sex'] = le.transform(data['Sex'].astype(str)) ## F : 0 , M: 1

le = LabelEncoder()
le.fit(data['Embarked'].astype(str))
data['Embarked'] = le.transform(data['Embarked'].astype(str)) ## C : 0 , Q: 1 , S : 2

### Train Test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test  = train_test_split(data.iloc[:,[0,2,3,4,5,6,7,8]], data.Survived, test_size = 0.25)


### CART : Gini 
from sklearn import tree
model = tree.DecisionTreeClassifier(max_depth = 3)
model = tree.DecisionTreeClassifier(max_depth = 4, criterion = 'entropy')
model.fit(X_train, Y_train)


model.score(X_train, Y_train)

###### Visualize Tree
from matplotlib import pyplot as plt
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
### pip install --upgrade sklearn
tree.plot_tree(model, feature_names = X_train.columns.tolist(),
               class_names = ["No", "Yes"],
               filled = True)
fig.savefig('ThirdTree_C50.png')

# Get Variable importance
importance = pd.DataFrame({"Feature" : X_train.columns.tolist(),
                           "Score" : model.feature_importances_})
importance.sort_values(by = "Score", ascending = False, inplace = True)
plt.barh( importance.Feature, importance.Score)

## Model Performance
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, accuracy_score, recall_score, precision_score
from sklearn.metrics import classification_report
### train 100%
pred_tr = model.predict(X_train)
predProb_tr = model.predict_proba(X_train)

confusion_matrix(Y_train, pred_tr)
accuracy_score(Y_train, pred_tr)
classification_report(Y_train, pred_tr)
recall_score(Y_train, pred_tr) # Sensitivity, TPR
precision_score(Y_train, pred_tr) # Specificity, TNR
roc_auc_score(Y_train, pred_tr)
fpr, tpr , _ = roc_curve(Y_train, predProb_tr[:,1])
plt.plot(fpr, tpr)


# test 70%
pred_ts = model.predict(X_test)
predProb_ts = model.predict_proba(X_test)
confusion_matrix(Y_test, pred_ts)

accuracy_score(Y_test, pred_ts)
recall_score(Y_test, pred_ts) # Sensitivity, TPR
precision_score(Y_test, pred_ts) # Specificity, TNR
roc_auc_score(Y_test, pred_ts)
fpr, tpr , _ = roc_curve(Y_test, predProb_ts[:,1])
plt.plot(fpr, tpr)



'''
test Error > training Erro ==> Overfitting
test Acc < training Acc > 5%

test Error < training Erro ==> underfitting
test Acc > training Acc 
3%, 5%

'''

####  Pruning : max_depth
depth =[]

acc_tr_CART =[]
acc_ts_CART = []

acc_tr_C50 =[]
acc_ts_C50 = []

for i in range(1,31):
    # CART Model
    dtree = tree.DecisionTreeClassifier(criterion = 'gini',max_depth = i)
    dtree.fit(X_train, Y_train)
    
    pred = dtree.predict(X_train)
    acc_tr_CART.append(accuracy_score(Y_train, pred))
    
    pred = dtree.predict(X_test)
    acc_ts_CART.append(accuracy_score(Y_test, pred))
    
    #### C50 Model
    dtree = tree.DecisionTreeClassifier(criterion = 'entropy',max_depth = i)
    dtree.fit(X_train, Y_train)
    pred = dtree.predict(X_train)
    acc_tr_C50.append(accuracy_score(Y_train, pred))
    
    pred = dtree.predict(X_test)
    acc_ts_C50.append(accuracy_score(Y_test, pred))

    
    depth.append(i)
    
    
Fitting = pd.DataFrame({"Depth" : depth,
                        "Acc_Train_CART" : acc_tr_CART,
                        "Acc_Test_CART": acc_ts_CART,
                        "Acc_Train_C50" : acc_tr_C50,
                        "Acc_Test_C50": acc_ts_C50})
    

X_cols=['''Rainfall, 'Soil_Fertility','Temp', 'Soit_WaterContent, 
       'Irrigation_Yes', 'Machine_Use_Yes','Natural_Disaster_Risk_High',
       'Natural_Disaster_Risk_Medium', 'Region_East',' Region_South',
       'Region_West', 'Fertilizer_Type_Chemical''']

    