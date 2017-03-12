# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 15:06:34 2016

@author: twshere
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib as jb
import pandas as pd
import numpy as np
c=predict
f=pd.read_csv(r"re/product_market.csv")
submit=f.loc[:,['市场名称映射值','农产品类别','农产品名称映射值','预测时间']]
submit=pd.concat((submit,c),axis=1)
submit.to_csv(r"submit.csv",index=False,header=False)
