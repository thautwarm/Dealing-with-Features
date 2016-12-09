# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 23:40:42 2016
CCF农产品价格预测数据预处理
@author: twshere
"""
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import time
unique=np.unique
DataFrame=pd.DataFrame

def first(ods,bigd):
     for i in ods.keys():
         if '时间' not in i and '价格' not in i:
             ods[i]=ods[i].map(lambda x:bigd[i][x])
     return ods


try:
    _begin_flag
except:
    f=pd.read_csv(r"re/farming.csv")
    ff=pd.read_csv(r"re/product_market.csv")
    ff.loc[ff.loc[:,'农产品市场所在省份']=='\ufeff新疆','农产品市场所在省份']='新疆'
    ff.columns=ff.columns.map(lambda x:"数据发布时间" if x=='预测时间' else x)
    f=f.loc[:,f.keys().difference({'规格','区域','数据入库时间','最低交易价格','最高交易价格'})];
    ff=ff.loc[:,ff.keys().difference({'规格','区域','数据入库时间','最低交易价格','最高交易价格','颜色','单位'})];
    f['数据发布时间']=f['数据发布时间'].map(lambda x:10e-7*time.mktime(time.strptime(x,"%Y-%m-%d")))
    ff['数据发布时间']=ff['数据发布时间'].map(lambda x:10e-7*time.mktime(time.strptime(x,"%Y-%m-%d")))
    ff.数据发布时间-=f.数据发布时间.min()
    f.数据发布时间-=f.数据发布时间.min()
    sp=[sorted(unique(f[i])) for i in f.keys()]
    dic=[dict(zip(['null']+i,range(len(i)+1))) for i in sp]
    bigd=dict(zip(f.keys(),dic))
    print(1)
    f=first(f,bigd);
    ff=first(ff,bigd);
    X=f.loc[:,f.keys().difference({ '最低交易价格', '平均交易价格', '最高交易价格','规格','区域','颜色','单位'})];
    X_test=ff.loc[:,ff.keys().difference({ '最低交易价格', '平均交易价格', '最高交易价格','规格','区域','颜色','单位'})]
    y=f.loc[:,'平均交易价格'];
    """
    y1=f.loc[:,'颜色'];
    rfc=RandomForestRegressor(n_estimators=100,n_jobs=2)
    print('deal fea1')
    rfc.fit(X,y1);
    X_test['颜色']=DataFrame(get_int_label(rfc.predict(X_test)))
    del rfc
    print('deal fea2')
    rfc=RandomForestRegressor(n_estimators=100,n_jobs=2)
    y2=f.loc[:,'单位']
    X['颜色']=y1;
    rfc.fit(X,y2);
    X_test['单位']=DataFrame(get_int_label(rfc.predict(X_test)))
    X['单位']=y2;
    del rfc
    y=f.loc[:,'平均交易价格'];
    _begin_flag=1
    
    X_test=np.array(X_test)
    X=np.array(X)
    y=np.array(y)
    joblib.dump(X,'X')
    joblib.dump(X_test,'X_test')
    joblib.dump(y,'y')
    """
if True:
    X_test=np.array(X_test)
    X=np.array(X)
    y=np.array(y)
    joblib.dump(X,'X_re')
    joblib.dump(X_test,'X_test_re')
    joblib.dump(y,'y_re')
    

