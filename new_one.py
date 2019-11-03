import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wordcloud
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

colors = ["#15ff00", "#ff0033"]

import seaborn as sns
sns.set_palette(colors)
sns.barplot(['Positive','Negative'],[positive,negative])
plt.show()




tweetLengthTrain = df['tweet'].str.len()

tweetLengthTest = df1['tweet'].str.len()

print(tweetLengthTrain, tweetLengthTest)
plt.hist(tweetLengthTrain,bins=20,label='Train_Tweet')
plt.hist(tweetLengthTest,bins=20,label='Test_Tweet')
plt.legend()
plt.show()

# df_sample = df.sample(1000)

# df1_sample = df1.sample(400)

# df_sample

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


#visualizing the data using wordcloud
#wordcloud takes string as an input, so we need to concatenate 
#all the tweets to get one complete string
from wordcloud import WordCloud
'''pending yet'''
words = ' '.join([token for token in combine['cleanedText']])
wordcloud = WordCloud().generate(words)
plt.figure(figsize = (10,10))
plt.imshow(wordcloud,interpolation="bilinear")
plt.show()



#function collecting hashtags
def hashtag(c):
    hashtags = []
    for i in c:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)
    return hashtags


hashtags_pos = hashtag(combine['cleanedText'][combine['label'] == 0])

hashtags_pos

#converted the list of lists of hashtags into one list(bag) 
hashtags_pos = sum(hashtags_pos,[])
hashtags_pos
# hashpos = sum(hashtags_pos)

#same for negative

hashtags_neg = hashtag(combine['cleanedText'][combine['label'] == 1])

hashtags_neg

#converted the list of lists of hashtags into one list(bag) 
hashtags_neg = sum(hashtags_neg,[])
hashtags_neg

#FreqDist = counts the number of occurences of each word in  the entire corpus
pos = nltk.FreqDist(hashtags_pos)
pos


neg = nltk.FreqDist(hashtags_neg)
neg


#to create a visual graph of the freq Dist

#create a dictionary to store the key(word) and its value(occurence)

listpos = list(pos.keys())
listcount = list(pos.values())
dic = pd.DataFrame({'Hashtag': listpos, 'Count': listcount})


#returns the first n rows ordered by columns in descending order
dic = dic.nlargest(columns ='Count', n=20)



axis = sns.barplot(data = dic, x = 'Hashtag', y='Count')
plt.figure(figsize = (16,5))

#assigns labels to x axis
plt.setp(axis.get_xticklabels(), rotation = 90)
plt.show()



#time for vectorizng the data(done playing with visualization)


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import gensim

from gensim.models import Word2Vec

#bagofwords feature extraction

bOW = CountVectorizer(stop_words = 'english',ngram_range=(1, 2))
print(bOW)
bagofwords = bOW.fit_transform(combine['cleanedText'])

# len(bagofwords.get_feature_names())
# output : 40696 (words in the corpus)

#tfidf feature extraction

# termidf = TfidfVectorizer(stop_words = 'english',ngram_range=(1, 1),max_features=10000)
# tfidf = termidf.fit_transform(combine['cleanedText'])
# # print(termidf.get_feature_names())
# tfidf



#Importing TFIDF 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
termidf = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 4), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')
tfidf = termidf.fit_transform(combine['cleanedText'])




from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report,f1_score

#apply algorithms

train_bow = bagofwords[:31962, :]
test_bow = bagofwords[31962: , :]

X_train,X_test,y_train,ytest = train_test_split(train_bow, df['label'], random_state=0, test_size=0.3)

logistic_model = LogisticRegression(random_state=42, solver ='saga')
logistic_model.fit(X_train,y_train)


y_pred = logistic_model.predict(X_test)
print("Logistic Regression using BOW",f1_score(ytest,y_pred))
print(classification_report(ytest, y_pred))
print(confusion_matrix(ytest, y_pred))


train_tfidf = tfidf[:31962, :]
test_tfidf = tfidf[31962: , :]

X_train,X_test,y_train,ytest = train_test_split(train_tfidf, df['label'], random_state=0, test_size=0.2)

logistic_model = LogisticRegression(random_state=42, solver ='saga')
logistic_model.fit(X_train,y_train)


y_pred = logistic_model.predict(X_test)
print("Logistic Regression using tfidf", f1_score(ytest,y_pred))
print(classification_report(ytest, y_pred))
print(confusion_matrix(ytest, y_pred))


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report,f1_score

#apply algorithms

train_bow = bagofwords[:31962, :]
test_bow = bagofwords[31962: , :]

X_train,X_test,y_train,ytest = train_test_split(train_bow, df['label'], random_state=0, test_size=0.3)

logistic_model = LogisticRegression(random_state=42, solver ='saga')
logistic_model.fit(X_train,y_train)


y_pred = logistic_model.predict(X_test)
print("Logistic Regression using BOW",f1_score(ytest,y_pred))
print(classification_report(ytest, y_pred))
print(confusion_matrix(ytest, y_pred))


train_tfidf = tfidf[:31962, :]
test_tfidf = tfidf[31962: , :]

X_train,X_test,y_train,ytest = train_test_split(train_tfidf, df['label'], random_state=0, test_size=0.2)

logistic_model = LogisticRegression(random_state=42, solver ='saga')
logistic_model.fit(X_train,y_train)


y_pred = logistic_model.predict(X_test)
print("Logistic Regression using tfidf", f1_score(ytest,y_pred))
print(classification_report(ytest, y_pred))
print(confusion_matrix(ytest, y_pred))

