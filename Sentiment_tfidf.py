import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import nltk
import re
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english')) - {'no', 'nor', 'not'}

import sys
print(sys.executable)



df = pd.read_csv('/home/ubuntu/Desktop/twitter_data/train.csv')
df1 = pd.read_csv('/home/ubuntu/Desktop/twitter_data/test.csv')

df.shape

print('Shape of Train Dataset ', df.shape)
print('Shape of Test Dataset', df1.shape)

# top 5 ids with positive labels
df[df['label'] == 0].head()

# top 5 ids with negative labels
df[df['label'] == 1].head()



positive = df['label'].value_counts()[0]
negative = df['label'].value_counts()[1]

#combining both train and test dataset to perfrom preprocessing 

combine  = df.append(df1, ignore_index = True)
print('Shape of new Dataset ', combine.shape)


def cleantext(input_w, pattern):
    r = re.findall(pattern = pattern, string  = input_w)
    for i in r:
        input_w = re.sub(pattern = i, repl ='', string = input_w)
    return input_w


#vectorize defines a vectorized function that takes numpy array as input
vect = np.vectorize(cleantext)
combine['cleanedText'] = vect(combine['tweet'],'@[\w]*')
combine.head()
#remove punctuation
combine['cleanedText'] = combine['cleanedText'].str.replace("[^a-zA-Z]"," ")
combine.head()


def remove_stopwords(text):
    return ' '.join([word for word in str(text).split() if word not in stopwords])

combine['cleanedText'] = combine['cleanedText'].apply(lambda text: remove_stopwords(text))

#remove words less than word length <3
combine['cleanedText'] = combine['cleanedText'].str.replace(r'\b(\w{1,2})\b','')
combine.head(100)

#time to tokenize_tweets
tokenized_tweets = combine['cleanedText'].apply(lambda x:x.split())

#this gives me tokens of each tweet
#now apply stemming on these tokens to get the root word of every token

from nltk.stem import PorterStemmer



#call object of that class
stemmer = PorterStemmer()

tokenized_tweets = tokenized_tweets.apply(lambda x: [stemmer.stem(i) for i in x])

tokenized_tweets.head()


#now join the tokens of each tweet to form a preprocessed tweet

for i in range(len(tokenized_tweets)):
    tokenized_tweets[i] = ' '.join(tokenized_tweets[i])
combine['cleanedText'] = tokenized_tweets

combine.head()



from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
#import gensim

#from gensim.models import Word2Vec




#Importing TFIDF 
#from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
termidf = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 4), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')
tfidf = termidf.fit_transform(combine['cleanedText'])




from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report,f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
#apply algorithms



train_tfidf = tfidf[:31962, :]
test_tfidf = tfidf[31962: , :]

X_train1,X_test1,y_train1,ytest1 = train_test_split(train_tfidf, df['label'], random_state=0, test_size=0.2)

logistic_model = LogisticRegression(random_state=42, solver ='saga')
logistic_model.fit(X_train1,y_train1)


y_pred1 = logistic_model.predict(X_test1)
print("Logistic Regression using tfidf", f1_score(ytest1,y_pred1))
print(classification_report(ytest1, y_pred1))
print(confusion_matrix(ytest1, y_pred1))


neigh = KNeighborsClassifier(n_neighbors = 2)
neigh.fit(X_train1, y_train1)

y_pred = neigh.predict(X_test1)
print("KNN using BOW",f1_score(ytest1,y_pred))
print(classification_report(ytest1, y_pred))
print(confusion_matrix(ytest1, y_pred))


svc = SVC()
svc.fit(X_train1, y_train1)

y_pred = svc.predict(X_test1)
print("SVC using BOW",f1_score(ytest1,y_pred))
print(classification_report(ytest1, y_pred))
print(confusion_matrix(ytest1, y_pred))
