from nltk.stem import *
import os
import math
import json
import pandas as pd 
import numpy
import pathlib

def preprocessing():
    path = pathlib.Path(__file__).parent.absolute()   
    stemmer = PorterStemmer()  
    path = str(path)
    #Loading file set to data frame
    df=pd.read_csv(path + "\\trac-gold-set\\agr_en_fb_gold.csv", encoding = "utf-8")
    
    #Case folding and porter stemming
    df["preprocess"]=df["text"].apply(lambda x:x.lower())
    df["preprocess"]=df["text"].apply(lambda x:stemmer.stem(x))
    print(df['preprocess'])

preprocessing()



        