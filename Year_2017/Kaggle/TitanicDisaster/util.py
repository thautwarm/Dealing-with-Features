# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 13:58:59 2017

@author: Thautwarm
"""
from __future__ import division
from functools import reduce
from itertools import repeat
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import math

import pandas as pd
def tfidf(wordarrs,labels,label,func=None):
    if not func:
        func=lambda x:x
    wordsPos=reduce(lambda x,y:x+y, (func(wordarr_i) for wordarr_i,label_i in zip(wordarrs,labels) if label_i ==label))
    wordsNeg=reduce(lambda x,y:x+y,  (func(wordarr_i) for wordarr_i,label_i in zip(wordarrs,labels) if label_i !=label))
    N_Neg=len(wordsNeg)
    N_Pos=len(wordsPos)
    wordsPosSet=set(wordsPos)
    result=[(word,wordsPos.count(word),wordsPos.count(word)/N_Pos,wordsNeg.count(word)/N_Neg) for word in wordsPosSet]
    return result
def getTfidfVec(result,thres=5):
    df=pd.DataFrame(result)
    df.columns=['word','num','tf','idf']
    df['vec']=(df.tf-df.idf)*df.num.map(lambda x:100+math.log10(x) if x >=thres else 0)
    return df

def strDig(x):
    return [len(i) if i.isdigit()  else i for i in x]
    
    
