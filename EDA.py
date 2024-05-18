import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
# nltk.download('stopwords')
#
# print(stopwords.words('english'))

df=pd.read_csv('training.csv', encoding='ISO-8859-1')
#Shape

df=pd.concat([df.iloc[:250000],df[800000:]], axis=0)
print(df.shape)
print(df.columns)
print(df.head())

new=pd.DataFrame()
new['sentiment']=df.iloc[:,0]
new['text']=df.iloc[:,5]
print(new.head())

# print(df.isnull().sum())

print(new['sentiment'].value_counts())

# new=df[df['sentiment'] != 'neutral']
# new.replace({'sentiment':{'negative':int(0), 'positive':int(1)}}, inplace=True)
# print(new['sentiment'].value_counts())


