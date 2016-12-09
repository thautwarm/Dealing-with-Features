#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 09:00:14 2016

@author: thaut
"""
#import pandas  as pd
import re
import numpy as np
from itertools import cycle 
#from matplotlib import pyplot as plt
from sklearn.preprocessing import Normalizer  as no
import random
from sklearn.cross_validation import KFold
def Range(x,n=0):
    return range(len(x)-n)

    
    
Seq=['A','C','G','U']
SeqD=[]
for i in Seq:
    SeqD+=list(zip(Seq,cycle(i)))
MapN=dict(zip(Seq,Range(Seq)))
MapD=dict(zip(SeqD,Range(SeqD)))
def GetFrequences(datas):

    pass
def rd(file):
    with open(file,'r') as f:
        ret=f.read()
    ret=re.findall('([A-Z]{9})[A-Z]{3}([A-Z]{9})',ret)
    for i in Range(ret):
        ret[i]=ret[i][0]+ret[i][1]
    return ret    
def rd0(file):
    with open(file,'r') as f:
        ret=f.read()
    ret=re.findall('([A-Z]{24})[A-Z]{3}([A-Z]{24})',ret)
    for i in Range(ret):
        ret[i]=ret[i][0]+ret[i][1]
    return ret
def Prior2(DataList):
    List1=DataList[:-1]
    List2=DataList[1:]
    List=list(zip(List1,List2))
    return List
def P2d(DataSets): #prior2 data
    Sets=list(map(Prior2,DataSets))
    return Sets
def transf(Datas,Map):
    F=lambda i:Map[i] 
    def sub_transform(l):
        l=list(map(F,l))
        return l
    return list(map(sub_transform,Datas))

def _Window(N_map,DataList,l=1,Amount=0):
    
    N=len(DataList)
    if type(l)==type(1):
        l=np.zeros((N_map,N))
        Amount=0
    for arg,kind in enumerate(DataList):
            l[kind,arg]+=1
    Amount+=1
    return l,Amount
def Window(Map,DataSets):
    N_map=len(Map)
    l=1;am=0
    for i in DataSets: 
        l,am=_Window(N_map,i,l,am)
    return l/am               
def _Process(Map,Data):
    x=transf(Data,Map)
    x=Window(Map,x)
    return x
def Process(Data,mode=1):
    if mode==1:
        x=P2d(Data)
        x=_Process(MapD,x)
    elif mode==0:
        x=_Process(MapN,Data)
    return x
    



def frequences_matrix_mainFunc(X_p,X_n):
    xp_1=Process(X_p)
    xn_1=Process(X_n)
    x_1=xp_1-xn_1
    x_1=no().fit_transform(x_1.T).T
    xp_0=Process(X_p,mode=0)
    xn_0=Process(X_n,mode=0)
    x_0=xp_0-xn_0
    x_0=no().fit_transform(x_0.T).T
    return x_0,x_1    
def _vec(dl,Table):
    am=len(dl)
    vec=np.zeros((am,))
    for arg,kind in enumerate(dl):
        vec[arg]+=Table[kind,arg]
    return vec
        
def copos_l(dl): #features from sequence coposition
    ret=np.zeros((4,))
    for i in dl:
        ret[MapN[i]]+=1
    ret/=len(dl)
    return ret
def Vectorize0(Datas,Table):
    mapn=lambda l:_vec(l,Table[0])
    mapd=lambda l:_vec(l,Table[1])
    d3=np.array(list(map(copos_l,Datas)))
    d1=P2d(Datas)
    d1=transf(d1,MapD)
    d1=np.array(list(map(mapd,d1)))
    d2=transf(Datas,MapN)
    d2=np.array(list(map(mapn,d2)))
    d=np.hstack((d2,d1,d3))
    return d

def Vectorize1(Datas,Table):
    d3=np.array(list(map(copos_l,Datas)))
    return d3
    
def Vectorize2(Datas,Table):
    mapn=lambda l:_vec(l,Table[0])
    d3=np.array(list(map(copos_l,Datas)))
    d2=transf(Datas,MapN)
    d2=np.array(list(map(mapn,d2)))
    d=np.hstack((d2,d3))
    return d

#define which features do you use
Vectorize=Vectorize2



def GetFeatures(X,y,Table):
    trn=X[y==0]
    trp=X[y==1]
    X_n=Vectorize(trn,Table)
    y_n=np.zeros((X_n.shape[0],))
    X_p=Vectorize(trp,Table)
    y_p=np.ones((X_p.shape[0],))
    X_ret=np.vstack((X_n,X_p))
    y_ret=np.hstack((y_n,y_p))
    return X_ret,y_ret
    
    

from sklearn.cross_validation import train_test_split as tts
from sklearn.grid_search import GridSearchCV as CV
from sklearn.metrics import roc_curve,auc
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.ensemble import VotingClassifier as votc
from math import sqrt
def classifierFunc(s):
    if s=='SVC':
        return SVC()
    elif s=='rf':
        return rf(n_estimators=50)
    elif s=='SVC':
        return SVC(kernel='linear')
    elif s=='rbf':
        return SVC(kernel='rbf',C=0.60,gamma=0.1675,probability=True)
def mixclass(alist):
        clf=[]
        vot=None
        for i in alist:
            for s in range(i[1]):
                sc=classifierFunc(i[0])
                clf.append(sc)
        num=len(clf)
        numarr=list(range(num))
        classifier=list(zip(numarr,clf))
        vot=votc(estimators=classifier,voting='soft',n_jobs=1+int(num/10))
        return vot
        
        
#initialize the datasets
try:
    trn
except:
    trn=rd0('DataSets/Met2614_N.fasta')
    trp=rd0('DataSets/Met2614_P.fasta')
    trall=np.array(trn+trp)
    y_n=np.zeros((len(trn),))
    y_p=np.ones( (len(trp),))
    yall=np.hstack((y_n,y_p))

#randomize the datasets
random_index=np.array(Range(trall))
random.shuffle(random_index)
trall=trall[random_index]
yall=yall[random_index]


def getmcc(y_pre,y_test):
    TP=sum((y_pre==1)&(y_test==1))
    TN=sum((y_pre==0)&(y_test==0))
    FP=sum((y_pre==1)&(y_test==0))
    FN=sum((y_pre==0)&(y_test==1))
    #print(TP,TN,FP,FN)
    MCC=(TP*TN-FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    Sen=TP/sum(y_test==1)
    Spe=TN/sum(y_test==0)
    return MCC,Sen,Spe
def getmcc2(model,x,y):
    y_pre=model.predict(x)
    return getmcc(y_pre,y)
#initialize the evaluating indicators
def stat(x):
   return np.mean(x),np.std(x)
def Print_mean(x,ch):
   x=np.mean(x)
   print(ch+':',x)
def main(): 
    kfold=KFold(len(yall),10)
    sen=[]  
    spe=[]
    acc=[]
    mcc=[] 
    figs=[]
    #set the params of SVM
    C=np.linspace(0.6,0.8,10)  
    G=np.linspace(0.13,0.22,10)
    clist=[]
    glist=[]
    aucs=[]
    param={'C':C,'gamma':G}
    for ind1,ind2 in kfold:
        print('*********')
        x_train=trall[ind1]
        y_train=yall[ind1] 
        X_p=x_train[y_train==1]
        X_n=x_train[y_train==0]
        Table=frequences_matrix_mainFunc(X_p,X_n)
        x_train,y_train=GetFeatures(x_train,y_train,Table)
        x_test=trall[ind2]
        y_test=yall[ind2]
        x_test,y_test=GetFeatures(x_test,y_test,Table)
        svm=SVC(kernel='rbf',probability=True)
        x1,x2,y1,y2=tts(x_train,y_train,test_size=0.2)
        cv=CV(svm,param,n_jobs=2)
        cv.fit(x2,y2)
        best=cv.best_params_
        c=best['C'];g=best['gamma']
        clist.append(c)
        glist.append(g) 
        print('c,g:',c,g)
        svm=SVC(kernel='rbf',C=c,gamma=g,probability=True)
        svm.fit(x_train,y_train)
        acc_r=svm.score(x_test,y_test)
        mcc_r,sen_r,spe_r=getmcc2(svm,x_test,y_test)
        acc.append(acc_r)
        mcc.append(mcc_r)
        sen.append(sen_r)
        spe.append(spe_r)
        scores=svm.predict_proba(x_test)[:,1]
        fpr,tpr,thres=roc_curve(y_test,scores)
        figs.append([fpr,tpr])
        #print('sen:',sen_r,'\n','spe:',spe_r)
        auc_r=auc(fpr,tpr)
        aucs.append(auc_r)
        print(auc_r)
        print('acc:',acc_r,'\n','mcc:',mcc_r)
        print('*********')
    return mcc,acc,aucs,sen,spe,figs
        
def repeat_mian():
    MCC=[]
    MCC_std=[]
    Sen=[]
    Sen_std=[]
    Spe=[]
    Spe_std=[]
    ACC=[]
    ACC_std=[]
    AUC=[]
    AUC_std=[]
    figs=[]
    for i in range(5):
            mcc,acc,aucs,sen,spe,figs_r=main()
            Print_mean(mcc,'mcc')
            Print_mean(acc,'acc')
            Print_mean(sen,'sen')
            Print_mean(spe,'spe')
            Print_mean(aucs,'auc')
            ACC.append(np.mean(acc))
            ACC_std.append(np.std(acc))
            MCC.append(np.mean(mcc))
            MCC_std.append(np.std(acc))
            Sen.append(np.mean(sen))
            Sen_std.append(np.std(sen))
            Spe.append(np.mean(spe))
            Spe_std.append(np.std(spe))
            AUC.append(np.mean(aucs))
            AUC_std.append(np.std(aucs))
            figs+=figs_r
    return MCC,ACC,AUC,Sen,Spe,figs




    