from model import SentimentAnalyzer

SAnalyzer=SentimentAnalyzer()
while(1):
    st=input("Enter string: ")
    SAnalyzer.predict(st)