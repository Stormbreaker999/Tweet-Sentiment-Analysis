import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
import nltk
import pickle

modelfile="model.sav"
vectorfile="vector.sav"
nltk.download('stopwords')
def stemming(content):
    content=str(content)
    stemmed_content=re.sub('[^a-zA-Z]',' ', content)
    stemmed_content=stemmed_content.lower()
    stemmed_content=stemmed_content.split()
    stemmed_content=[port_stem.stem(word) for word in stemmed_content]
    stemmed_content=' '.join(stemmed_content)
    return stemmed_content

df=pd.read_csv('training.csv', encoding='ISO-8859-1')
df=pd.concat([df.iloc[:280000],df[800000:]], axis=0)
new=pd.DataFrame()
new['sentiment']=df.iloc[:,0]
new['text']=df.iloc[:,5]

new.replace({'sentiment':{4:1}}, inplace=True)

port_stem=PorterStemmer()
new['stemmed_text'] = new['text'].apply(stemming)

new = new[new['stemmed_text'] != 'nan']
X = new['stemmed_text']
Y = new['sentiment']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=2)

# Vectorizing
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

def model_design():

    model=LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)

    #Accuracy score
    X_train_prediction=model.predict(X_train)
    train_accuracy=accuracy_score(Y_train, X_train_prediction)

    X_test_prediction=model.predict(X_test)
    test_accuracy=accuracy_score(Y_test, X_test_prediction)

    print(train_accuracy, test_accuracy)
    pickle.dump(vectorizer,open(vectorfile, "wb"))
    return model


class SentimentAnalyzer:
    def __init__(self):
        if not os.path.isfile("model.sav"):
            self.model=model_design()
            pickle.dump(self.model, open(modelfile, "wb"))
        else:
            train_choice=input("Want to train again(y/n)")
            if train_choice=='y':
                self.model = model_design()
                pickle.dump(self.model, open(modelfile, "wb"))
            else:
                self.model=pickle.load(open(modelfile, "rb"))
    def predict(self, query):
        vectorizer=pickle.load(open(vectorfile,"rb"))
        feature = np.array([query])
        feature = vectorizer.transform(feature)
        pred = self.model.predict(feature)
        if int(pred[0]):
            print("Positive Thought")
        else:
            print("Negative Thought")

