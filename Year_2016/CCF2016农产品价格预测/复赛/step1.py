
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
    f=pd.read_csv(r"farming.csv")
    ff=pd.read_csv(r"product_market.csv")
    ff.loc[ff.loc[:,'农产品市场所在省份']=='\ufeff新疆','农产品市场所在省份']='新疆'
    ff.columns=ff.columns.map(lambda x:"数据发布时间" if x=='预测时间' else x)
    f=f.loc[:,f.keys().difference({'规格','区域','数据入库时间','最低交易价格','最高交易价格','颜色','单位'})];
    ff=ff.loc[:,ff.keys().difference({'规格','区域','数据入库时间','最低交易价格','最高交易价格','颜色','单位'})];
    f['数据发布时间']=f['数据发布时间'].map(lambda x:time.mktime(time.strptime(x,"%Y-%m-%d")))
    ff['数据发布时间']=ff['数据发布时间'].map(lambda x:time.mktime(time.strptime(x,"%Y-%m-%d")))
    ff.数据发布时间/=3600
    f.数据发布时间/=3600
    ff.数据发布时间-=f.数据发布时间.min()
    f.数据发布时间-=f.数据发布时间.min()


    keys=sorted(f.keys().difference({'数据发布时间','平均交易价格'}))
    sp=[sorted(set(f[i])) for i in keys]
    dic=[dict(zip(['null']+i,range(len(i)+1))) for i in sp]
    bigd=dict(zip(keys,dic))
    f=first(f,bigd);
    ff=first(ff,bigd);
    X=f.loc[:,f.keys().difference({ '最低交易价格', '平均交易价格', '最高交易价格','规格','区域'})];
    X_test=ff.loc[:,ff.keys().difference({ '最低交易价格', '平均交易价格', '最高交易价格','规格','区域'})]
    y=f.loc[:,'平均交易价格'];
    X.to_csv('train.csv',index=False)
    X_test.to_csv('test.csv',index=False)
    y.to_csv('y.csv',index=False)

    
