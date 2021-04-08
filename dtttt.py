# -*- coding: utf-8 -*-
"""
Created on Tue May 26 19:03:38 2020

@author: kisho
"""
import pandas as pd
df=pd.read_csv("placement.csv")
df.isna().sum()

#fill the missing value
df["salary"].fillna(df.salary.mean(skipna=True),inplace=True)
df.isna().sum()
del df["sl_no"]

#label encoder
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
le.fit(df["gender"].astype(str))
df["gender"]= le.transform(df["gender"].astype(str))

le=LabelEncoder()
le.fit(df["ssc_b"].astype(str))
df["ssc_b"]= le.transform(df["ssc_b"].astype(str))

le=LabelEncoder()
le.fit(df["hsc_b"].astype(str))
df["hsc_b"]=le.transform(df["hsc_b"].astype(str))

le=LabelEncoder()
le.fit(df["hsc_s"].astype(str))
df["hsc_s"]=le.transform(df["hsc_s"].astype(str))

le=LabelEncoder()
le.fit(df["status"].astype(str))
df["status"]=le.transform(df["status"].astype(str))

le=LabelEncoder()
le.fit(df["degree_t"].astype(str))
df["degree_t"]=le.transform(df["degree_t"].astype(str))

le=LabelEncoder()
le.fit(df["workex"].astype(str))
df["workex"]=le.transform(df["workex"].astype(str))

le=LabelEncoder()
le.fit(df["specialisation"].astype(str))
df["specialisation"]=le.transform(df["specialisation"].astype(str))

#mode
df.ssc_b.value_counts()
df.degree_t .value_counts()

