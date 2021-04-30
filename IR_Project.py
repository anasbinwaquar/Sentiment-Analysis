from nltk.stem import *
import os
import math
import json
import pandas as pd 
import numpy
import re
import pathlib
import textstat
from collections import defaultdict

def flesch_reading_ease(text):
    # formula=206.835-1.015(total_words/1)-84.6(syllables/total_words)
    syllables=textstat.syllable_count(text)
    words=textstat.lexicon_count(text, removepunct=True)
    score=206.835-1.015*(words)-84.6*(syllables/words)
    # print(score)
    return score

def flesch_kincaid_grade_level(text):
    # formula=0.39*(total_words/1)+11.8(syllables/total_words)-15.59
    syllables=textstat.syllable_count(text)
    words=textstat.lexicon_count(text, removepunct=True)
    score=0.39*(words/1)+11.8*(syllables/words)-15.59
    # print(score)
    return score

def total_characters(text):
    count=0
    for char in text:
        count += 1
    return count

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
    df["flesch_reading_ease"]=df["text"].apply(lambda x:flesch_reading_ease(x))
    df["flesch_kincaid_grade_level"]=df["text"].apply(lambda x:flesch_kincaid_grade_level(x))
    df["syllables"]=df["text"].apply(lambda x:textstat.syllable_count(x))
    df["words"]=df["text"].apply(lambda x:textstat.lexicon_count(x))
    df["characters"]=df["text"].apply(lambda x:total_characters(x))

    #Export processed CSV
    df.to_csv('processed_data.csv')

preprocessing()
# print(flesch_reading_ease('hello this is mate'))



        