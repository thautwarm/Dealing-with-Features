# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 13:48:17 2017

@author: Thautwarm
"""
from DataRead import TestDatas,TrainDatas
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from functools import reduce
from util import tfidf,getTfidfVec,strDig
cut=lambda x:word_tokenize(x,language='english')
#splitName=TrainDatas.Name.map(cut)
Features=[]
model=[]
modelName=[]
arr=reduce(lambda x,y:x+y,TrainDatas.Name)
asciis=''.join(set(arr.lower())-set([chr(ord('a')+i) for i in range(26)]))
resub_Name=lambda x:re.sub(r'[^a-zA-Z]',' ',x)
wordsFromName=TrainDatas.Name.map(lambda x:cut(resub_Name(x)))
df_Name=getTfidfVec(tfidf(wordsFromName,TrainDatas.Survived,1))
mapping_Name=dict(zip(df_Name.loc[df_Name.num>5].word,df_Name.loc[df_Name.num>5].vec))
model.append( lambda x:sum( [mapping_Name[key] for key in mapping_Name if  key in cut(resub_Name(x)) ]) )
#df_Name.loc[df_Name.num<5,'vec']=0
modelName.append('Name')


resub_Ticket=lambda x:re.sub(r'[^0-9]','',x)
wordsFromTicket=TrainDatas.Ticket.map(lambda x:cut(resub_Ticket(x)))
df_Ticket=getTfidfVec(tfidf(wordsFromTicket,TrainDatas.Survived,1,func=strDig),thres=10)
mapping_Ticket=dict(zip( df_Ticket.loc[df_Ticket.num>=10].word,df_Ticket.loc[df_Ticket.num>=10].vec.astype(int)))
model.append( lambda x:sum( [mapping_Ticket[key] for key in mapping_Ticket if  key in strDig(cut(resub_Ticket(x))) ]) )
#df_Ticket.loc[df_Ticket.num<10,'vec']=0
modelName.append('Ticket')
#Ticket_NumLength==4,5,6

Sex=lambda x: x=='female'
model.append(Sex)
modelName.append('Sex')

Cabin=lambda x: x==x
model.append(Cabin)
modelName.append('Cabin')


Age=lambda x: 1+int(x/2) if x==x else 0
model.append(Age)
modelName.append('Age')

Fare=lambda x: int(x/2)+1 if x==x else 0
model.append(Fare)
modelName.append('Fare')


def Embarked(x):
    if x=='Q':
        return 1
    elif x=='C':
        return 2
    elif x=='S':
        return 3
    else:
        return 0
model.append(Embarked)
modelName.append('Embarked')

def someTrivial(df):
    ret=(df.SibSp+df.Parch)
    return ret
tr= lambda D: pd.concat( [ D[key].map(func) for key,func in zip(modelName,model)]+[someTrivial(D)], axis=1)
def trans2(df):
    for i in {-15.0, -3.0, 0.0, 22.0}:
        df['Tic%d'%(int(i))]=df['Ticket']==i
    del df['Ticket']
    for i  in {0.0, 1.0, 2.0, 3.0}:
        df['Embarked%d'%(int(i))]=df['Embarked']==i
    del df['Embarked']
    df['Name']/=10
    return df

X_test= trans2(tr(TestDatas)).astype(float).values
X_train=trans2(tr(TrainDatas)).astype(float).values
y_train=TrainDatas.Survived.values

#Sex








