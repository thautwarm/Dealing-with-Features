# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 15:59:21 2017

@author: Thautwarm
"""

from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split 
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.naive_bayes import GaussianNB as nb
from sklearn.ensemble import GradientBoostingClassifier as gbdt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.lda import LDA
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import VotingClassifier as vot
import warnings
warnings.filterwarnings("ignore")
#from keras.models import Sequential
#from keras.layers import Dense,Dropout
#from keras.utils.np_utils import to_categorical
from sklearn.grid_search import GridSearchCV
import numpy as np
from logicMain import X_test,X_train,y_train,TestDatas
import pandas as pd
from functools import reduce
def stats(y_true,y_pred):
    TP=(y_true==1)&(y_pred==1)
    TN=(y_true==0)&(y_pred==0)
    FP=(y_true==0)&(y_pred==1)
    FN=(y_true==1)&(y_pred==0)
    tp=sum(TP)
    fp=sum(FP)
    fn=sum(FN)
    acc=(sum(TP)+sum(TN))/len(y_true)
    pre=0 if fp+tp==0 else tp/(tp+fp) 
    rec=0 if fn+tp==0 else tp/(tp+fn)
    f1= 0 if pre==0 or  rec==0 else 2/(1/pre +1/rec)
    select={'TP':TP,'TN':TN,'FP':FP,'FN':FN,'acc':acc}    
    return  pre,rec,f1,select
#net=Sequential()
#net.add(Dense(40,input_dim=X_train.shape[1],init='uniform',activation='linear'))
#net.add(Dense(40,input_dim=40,activation='relu'))
#net.add(Dense(40,input_dim=40,activation='relu'))
#net.add(Dropout(0.3))
#net.add(Dense(30,input_dim=40,activation='tanh'))
#net.add(Dense(1,input_dim=30,activation='relu'))
#net.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
#net = Sequential()
#net.add(Dense(64, input_dim=X.shape[1], init='uniform', activation='relu'))
#net.add(Dropout(0.5))
#net.add(Dense(64, activation='relu'))
#net.add(Dropout(0.5))
#net.add(Dense(1, activation='sigmoid'))
#net.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

Predictors=[]
N=len(X_train)
while len(Predictors)<=100:
    index=np.random.permutation(N)
    X_train=X_train[index]
    y_train=y_train[index]
    kf=KFold(len(X_train),n_folds=6)
    for train_index,test_index in kf:
        x1,x2=X_train[train_index],X_train[test_index]
        y1,y2=y_train[train_index],y_train[test_index]
        clf=vot(estimators=[
                    ('rfc',rfc(n_estimators=500)),
                    ('gbdt',gbdt(n_estimators=500,learning_rate=0.02))
                    ] ,weights=[5,5])
        clf.fit(x1,y1)
        y_pred=clf.predict(x2)
        pre,rec,f1,select=stats(y2,y_pred)
        print()
        print(" precision :%.4f recall : %.4f f1-score :%.4f "%(pre,rec,f1))
        if pre>0.83 and rec>0.72:
            Predictors.append(clf)
y=(np.mean([clf_i.predict(X_test) for clf_i in Predictors],0)>0.5)
y=pd.Series(y,name='Survived')
df=pd.DataFrame([TestDatas.PassengerId,y]).T
df.to_csv('res.csv',index=False)
    #{'gamma': 0.26722222222222219, 'C': 0.55555555555555558}

