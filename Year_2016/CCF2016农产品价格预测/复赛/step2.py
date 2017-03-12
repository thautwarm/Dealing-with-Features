#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 21:02:46 2016

@author: thaut
"""
from sklearn.preprocessing import OneHotEncoder
def div(c,duang):
    ret=[]
    label=[]
    table=c[duang]
    ch=table[0]
    t=0;
    for i in range(c.shape[0]):
        if table[i]!=ch:
            label.append(ch)
            ch=table[i]
            ret.append((c.iloc[t:i,:].copy()))
            
            t=i
        else:
            continue
    ret.append(c.iloc[t:i,:])
    label.append(ch)
    return ret,label
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor as rf
unique=np.unique
DataFrame=pd.DataFrame
def first(ods,bigd):
     for i in ods.keys():
         if '时间' not in i and '价格' not in i:
             ods[i]=ods[i].map(lambda x:bigd[i][x])
     return ods
try:
    step2
except:
    X=pd.read_csv('train.csv')
    getindex=lambda x:range(len(x.index))
    
    X_test=pd.read_csv('test.csv')
    
    y=pd.read_csv('y.csv',header=None)
    y.columns=['jiage']
    Xtrain=X
    X=pd.concat((X,y),axis=1)
    

        
    
