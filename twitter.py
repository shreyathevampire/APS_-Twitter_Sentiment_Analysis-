#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing data

train = pd.read_csv('/home/ubuntu/Downloads/nlp/train.csv')
test = pd.read_csv('/home/ubuntu/Downloads/nlp/test_tweets.csv')

#working on only a sample of data right now
train = train.sample(2000)
test = test.sample(500)


#preprocessing begins on datasize of 2000 tweets from here
#need to remove twitter handles of users which can be done by using re
import re
#pattern to be removed "@[\w]"
def remove_pattern(input , str):
    r = re.findall(str,input)
    for i in r:
        input = re.sub(i,'',str)
    return input

#converts the the data into ndarray that can be used later for vectorization purpose
train['tweet'] = np.vectorize(remove_pattern)(train['tweet'], "@[\w]*")
test['tweet'] = np.vectorize(remove_pattern)(test['tweet'], "@[\w]*")


#replace all punctuations,numbers
train['tweet'] = train['tweet'].str.replace("[^a-zA-Z#]"," ")
test['tweet'] = test['tweet'].str.replace("[^a-zA-Z#]"," ")
#replace all hashtags with a space
train['tweet'] = train['tweet'].str.replace("#[\w]*"," ")
test['tweet'] = test['tweet'].str.replace("#[\w]*"," ")




#replace all hashtags with a space
print(train['tweet'].replace(r'^\s*$', np.nan, regex=True))
# test['tweet'] = test['tweet'].str.replace("#[\w]*"," ")



#remove short words from the sentences (ie remove word length less than 3)
#words such as "ohh","hmm"
#these words add no meaning to the sentence
#func: .apply takes a function and applies it to all values of pandas series

train['tweet'] = train['tweet'].apply(lambda x : ' '.join([w for w in x.split() if len(w)>3]))

test['tweet'] = test['tweet'].apply(lambda x : ' '.join([w for w in x.split() if len(w)>3]))


#replace empty strings in tweet with NaN
train['tweet'].replace('', np.nan, inplace=True)

#drop rows containing NaN
train.dropna(subset=['tweet'], inplace=True)



#tokenize the data
tokenized_tweet = train['tweet'].apply(lambda x : x.split())

tokenized_tweet.head


#stemming needs to be performed
from nltk.stem.porter import PorterStemmer
#importing the object from Stemmer class
stemmer = PorterStemmer()
#stemming
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) 




train['tweet'] = tokenized_tweet

train.head()









