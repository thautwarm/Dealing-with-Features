# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 13:43:36 2017

@author: Thautwarm
"""

import os
pathNow=os.getcwd()
dataPath=pathNow.replace('H:','I:')

import pandas as pd
path=lambda x:"%s\\%s"%(dataPath,x)
TrainDatas=pd.read_csv(path('train.csv'),encoding='utf-8')
TestDatas=pd.read_csv(path('test.csv'),encoding='utf-8')
