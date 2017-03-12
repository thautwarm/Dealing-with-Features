import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.svm import SVR
from sklearn.cross_validation import train_test_split as tts
from sklearn.ensemble import RandomForestRegressor as rf
from sklearn.preprocessing import OneHotEncoder as ohe
from sklearn.neighbors import  KNeighborsRegressor as KNN
from sklearn.linear_model import  LinearRegression as LR
from sklearn.linear_model import  LogisticRegression as  LoR
from sklearn.linear_model import BayesianRidge as nb
from sklearn.linear_model import RidgeCV as rr
def TimeSeriesEX(T,n=-1):
    TT=[]
    num=T.shape[0]
    
    if n==-1:
        n=min(num//2,10)
    for i in range(n):
        subt=T[i:num-n+i]
        subt.index=T.index[:num-n]
        TT.append(subt)
    TT=pd.concat(TT,axis=1)
    return TT
    
    
    
def div(c,duang):
    ret=[]
    table=c[duang]
    ch=table[0]
    t=0;
    for i in range(c.shape[0]):
        if table[i]!=ch:
            ch=table[i]
            ret.append(c.iloc[t:i,:].copy(deep=True))
            t=i
        else:
            continue
    ret.append(c.iloc[t:i,:])
    return ret
def search(c,duang1,i1,duang2,i2):
    c1=c.loc[(c[duang1]==i1)&(c[duang2])==i2,:];
    return c1.std(),c1.mean()
    
def Split(train,size=0.5):
    len1=train.shape[0]
    check=-1.0*np.ones(len1,)
    check=np.isnan(check)
    i=0;
    split1=[]
    split2=[]
    keys=['农产品名称映射值']
    keys_all=train.keys()[:-2]
    while (i<len1):
        if check[i]==True:
            i+=1
            continue
        print(i)
        values=train.loc[i,:][:-2];
        ind = np.zeros((len1,))
        ind = ~np.isnan(ind)
        
        
        
        
        for arg,key in enumerate(keys):
            ind&=train[key]==values[arg]
        """
        ind_all = np.zeros((len1,))
        ind_all = ~np.isnan(ind_all)
        for arg,key in enumerate(keys_all):
            ind_all&=train[key]==values[arg]
        ind1.append(ind_all)
        """
        X=train.loc[ind,:].copy()
        X.sort_values(by='数据发布时间',inplace=True)
        """
        subX=X.loc[:,X.keys().difference({'数据发布时间','jiage'})]
        subX=to_categorical(subX.values)
        T=X.loc[:,('数据发布时间','jiage')].values
        subX=np.hstack((subX,T))
        """
        """
        tool=ohe(sparse=False)
        X1=tool.fit_transform(X.loc[:,X.keys()-{'数据发布时间','jiage','农产品名称映射值'}])
        X=np.hstack((X1,X.loc[:,('数据发布时间','jiage')].values))
        """
        n1=int(X.shape[0]*size)
        X.sort_values(by='数据发布时间',inplace=True)
        x1=X.iloc[:n1,:]
        x2=X.iloc[n1:,:]
        split1.append(x1)
        split2.append(x2)
        check[ind]=True
        i+=1
        if i>3000:break
        x1=pd.concat(split1)
        x2=pd.concat(split2)
        Xtr=x1.iloc[:,:-1]
        Xte=x2.iloc[:,:-1]
        ytr=x1.iloc[:,-1]
        yte=x2.iloc[:,-1]
    return Xtr,Xte,ytr,yte
    
    
def func(test,train):
    len1=test.shape[0]
    check=pd.Series(-1.0*np.ones(len1,))
    i=0;
    keys=train.keys()
    ch='农产品名称映射值'
    while(i<len1):
        if check[i]>0:
            i+=1
            continue
        
        dif={ch}
        ind=(test[ch]==test.loc[i,ch])
        ts=test.loc[ind,keys.difference(dif)]
        split=train.loc[(train[ch]==test.loc[i,ch]),keys.difference(dif)]
        dic=ts.keys().difference({'jiage'}.union(dif))       
        """
        X_new=split.loc[:,split.keys().difference(dif.union({'jiage'}))]
        Xt_new=test.loc[ind,test.keys().difference(dif)]
        """
        #if len(set(ts['农产品市场所在省份']))>=1:
        """
        tool=ohe(sparse=False)
        n=split.shape[0]
        X=pd.concat((split.loc[:,dic],ts.loc[:,dic]))
        X=tool.fit_transform(X)
        X_new=pd.DataFrame(X[:n])
        X_new.index=split.index
        Xt_new=pd.DataFrame(X[n:])
        Xt_new.index=ts.index           
        X_new=pd.concat((X_new,split['数据发布时间']),axis=1)
        Xt_new=pd.concat((Xt_new,ts['数据发布时间']),axis=1)
        """
        X_new=split.loc[:,dic]
        Xt_new=ts.loc[:,dic]      
        print(i,X_new.shape)
        this=rf(n_estimators=50)
        y=split.loc[:,'jiage']
        this.fit(X_new,y)
        y_pre=this.predict(Xt_new)
        check[ind]=y_pre
        #print("pre:",y_pre)
        i+=1
    return check


from sklearn import naive_bayes
def valitr(xtrain,xtest):
    ytest=xtest.loc[:,'jiage']
    white=boosttrain(xtest.loc[:,xtest.keys().difference({'jiage'})],xtrain)
    return np.sum(abs(white-ytest))
#0:mix 1:svc 2:LR 3:rf
predict=func(X_test,X)

"""
rfc=rf(n_estimators=50)
rfc.fit(X_train,y_train)
y1=rfc.predict(X_test)
print(rfc.score(X_test,y_test))
print(np.std(y1-y_test))
"""
