# Practice Problem : Twitter Sentiment Analysis by Analytics Vidhya 
#Author - Sachin Kumar

#Importing Libraries 
import re #cleaning the text 
import pandas as pd 
import numpy as np 
import string
import nltk
import warnings 

#Importing dataset
dataset = pd.read_csv('train.csv')
testdata = pd.read_csv('test.csv')

#combine train and test set
combi = dataset.append(testdata, ignore_index=True)

## importing regular expression library ## clean tweet text by removing links, special characters etc
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt

# remove twitter handles (@user)
combi['tweet'] = np.vectorize(remove_pattern)(combi['tweet'], "@[\w]*")

# remove special characters, numbers, punctuations
combi['tweet'] = combi['tweet'].str.replace("[^a-zA-Z#]", " ")
       
#Removing Short Words       
combi['tweet'] = combi['tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))       

tokenized_tweet = combi['tweet'].apply(lambda x: x.split())


from nltk.stem.porter import *
stemmer = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming

#Now let’s stitch these tokens
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

combi['tweet'] = tokenized_tweet

#Bag-of-Words Features

from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# bag-of-words feature matrix
bow = bow_vectorizer.fit_transform(combi['tweet'])


from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

train_bow = bow[:31962,:]
test_bow = bow[31962:,:]

# splitting data into training and validation set
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, dataset['label'], random_state=42, test_size=0.3)

from sklearn.svm import SVC
svm = SVC()
svm = SVC(kernel = 'rbf', random_state = 0, gamma = 0.14, C =11)
svm.fit(xtrain_bow, ytrain)
y_pred5 = svm.predict(xvalid_bow)
prediction_int7 = y_pred5.astype(np.int)
f1_score(yvalid, prediction_int7)

#prediction on test set
test_pred = svm.predict(test_bow)
test_pred_int = test_pred.astype(np.int)
testdata['label'] = test_pred_int
submission = testdata[['id','label']]
submission.to_csv('svmrbfbow.csv', index=False) # writing data to a CSV file