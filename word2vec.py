#importing the word2vec dictionary stored on drive

from google.colab import drive
drive.mount('/gdrive')
%cd /gdrive


#importing libraries
import pandas as pd
import numpy as np
import re
from sklearn import svm
from sklearn.model_selection import train_test_split


#preprocessing of text

def cleantext(input_w, pattern):
    r = re.findall(pattern = pattern, string  = input_w)
    for i in r:
        input_w = re.sub(pattern = i, repl ='', string = input_w)
    return input_w


df = pd.read_csv('/gdrive/My Drive/train.csv')
df1_orig = pd.read_csv('/gdrive/My Drive/test.csv')

df1 = df1_orig.sample(n = 1000, random_state = 42)



#vectoring the data

vect = np.vectorize(cleantext)
fake = df.loc[df['label'] == 1]
#print("fake = ", fake.shape)
#combining both train and test dataset to perfrom preprocessing 
non_fake = df.loc[df['label'] == 0].sample(n = 2000, random_state = 42)
#print("non-fake = " , non_fake.shape)
normalized_df = pd.concat([fake, non_fake])
#normalized_df.head()
#print(normalized_df.shape)
combine  = normalized_df
combine['cleanedText'] = vect(combine['tweet'],'@[\w]*')
combine.head()
#remove punctuation
combine['cleanedText'] = combine['cleanedText'].str.replace("[^a-zA-Z]"," ")
combine.head()

#remove words less than word length <3
combine['cleanedText'] = combine['cleanedText'].str.replace(r'\b(\w{1,2})\b','')
combine.head()
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

label = df['label']


combine.head()
tweet  =  list((combine['cleanedText']))

#reading the dictionary
file = '/gdrive/My Drive/glove.twitter.27B.200d.txt'
d= {}
glove = open(file,"r+")
for line in glove:
    values = line.split()
    word = values[0]
    vector = np.asarray(values[1:], dtype='float32')
    d[word] = vector
glove.close()
vec = np.zeros([label.shape[0],200], dtype = 'float32') 
c=0
for i in tweet:
    for j in i:
        try:
            j=str(j)
            k=d[j]
            vec[c]=(vec[c]+np.array(k))
        except:
            continue
    c=c+1
    
#splitting the data into train and test set
X_train,X_test,y_train,y_test= train_test_split(vec,label,test_size=0.1,random_state=42)



#creating SVM classifier

clf=svm.SVC()
clf.fit(X_train,y_train)
prediction=clf.predict(X_test)
pred_labels=prediction==y_test
acc=0.0
for i in pred_labels:
    if i==True:
        acc+=1
print ('SVM accuracy=',(acc)/len(pred_labels)*100,'%')

#creating KNN Classifier

from sklearn.neighbors import KNeighborsClassifier
clf =KNeighborsClassifier()         
clf.fit(X_train,y_train)
prediction=clf.predict(X_test)
pred_labels=prediction==y_test
acc=0.0
for i in pred_labels:
    if i==True:
        acc+=1
print ('KNN accuracy=',(acc)/len(pred_labels)*100,'%')


#creating LogisticRegression Classifier

from sklearn.linear_model import LogisticRegression
clf =LogisticRegression(random_state=42, solver ='saga')         
clf.fit(X_train,y_train)
prediction=clf.predict(X_test)
pred_labels=prediction==y_test
acc=0.0
for i in pred_labels:
    if i==True:
        acc+=1
print ('Logistic accuracy=',(acc)/len(pred_labels)*100,'%')
