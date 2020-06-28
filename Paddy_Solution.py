
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 13:17:55 2020

@author: ADMIN
"""


######### MLRM Model - Paddy Crop Yield
import pandas as pd
df = pd.read_csv("Linear_Regression/PaddyCropYield.csv")
df.describe()
df.dtypes
df.isna().sum()

### Visualiza Data
import matplotlib.pyplot as plt
for col in df.columns.tolist():
    if (df[col].dtype.name == 'float64'):
        plt.hist(df[col].dropna())
        plt.savefig("Linear_Regression/Paddy_Plots/hist_"+col)
        plt.close()
        
        plt.scatter(df[col], df.Yield)
        plt.savefig("Linear_Regression/Paddy_Plots/scatter_"+col)
        plt.close()

        
    else:
        plt.barh(df[col].value_counts().index.tolist(), df[col].value_counts().values.tolist())
        plt.savefig("Linear_Regression/Paddy_Plots/bar_"+col)
        plt.close()
        
        plt.scatter(df[col].astype('str'), df.Yield)
        plt.savefig("Linear_Regression/Paddy_Plots/scatter_"+col)
        plt.close()



#### Impute Missing values
for col in df.columns.tolist():
    if (df[col].dtypes.name != "object"):
        df[col].fillna(df[col].median(skipna=True), inplace = True) 
    else:
        rep = df[col].value_counts().index[0]
        df[col].fillna(rep, inplace = True) 

df.isna().sum()
df.dtypes




##### Encode Categorical Variables
categorical_dist = pd.DataFrame(columns=['Variable','Least_Label'])
for col in df.columns.tolist():
    if (df[col].dtypes.name == "object"):
        df = df.join(pd.get_dummies(df[col], prefix=col))
        df2 = pd.DataFrame({'Variable':[col], 'Least_Label' : [df[col].value_counts().index[-1]]})
        categorical_dist = categorical_dist.append(df2, ignore_index = True)
        df.drop(col, axis = 1, inplace = True)
        
        
        
        

#### Train test split
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df)
      
cols_removed = categorical_dist["Variable"] + "_"+ categorical_dist["Least_Label"]
x_cols = set(df.columns).difference(set(cols_removed)).difference(set(["Yield"]))



### Multicollinearity treatment
# step :1
x_cols = x_cols.difference(set(["Machine_Use_No"]))
# step:2
x_cols = x_cols.difference(set(["Soit_WaterContent"]))

### After Intercept : 
#step3:
x_cols = x_cols.difference(set(["Machine_Use_Yes"]))
## step4:
x_cols = x_cols.difference(set(["Irrigation_No"]))

### t-test
x_cols = x_cols.difference(set(["Region_East",
                                "Region_North", "Region_West","Natural_Disaster_Risk_Low",   "Fertilizer_Type_Organic"]))
## step 2:
x_cols = x_cols.difference(set(["Natural_Disaster_Risk_Medium",
                                "Region_South", "Fertilizer_Type_Chemical"]))


### Remove Intercept - Add k labels
x_cols = x_cols.union(cols_removed)

formula = "Yield ~ " + " + ".join(x_cols) + "-1"


import statsmodels.formula.api as smf
reg = smf.ols(formula, data = train_df).fit()

reg.summary()
### Diagnostic Checks
import matplotlib.pyplot as plt
### Scatterplot Residual vs Fitted : Linearity Assumption
plt.scatter(reg.fittedvalues, reg.resid)

## Normality Assumption
from statsmodels.graphics.gofplots import qqplot
qqplot(reg.resid, line='s')

### Homoscedasticity Assumption
plt.scatter(reg.fittedvalues, reg.get_influence().resid_studentized_internal)

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(train_df[x_cols].values, i) for i in range(train_df[x_cols].values.shape[1])]
vif["feaures"] = x_cols
df.corr()["Yield"].sort_values(ascending = False)

#### save model
import pickle
filename = 'final_Paddy_MLRM.sav'
pickle.dump(reg, open(filename, 'wb'))

### Load model
model_loaded = pickle.load(open(filename, "rb"))
