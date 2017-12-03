# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', 
                      delimiter = '\t', #Tab Separated File format recommended for text analysis
                      quoting =3) # Ignoring double quotes

# Clean the texts and create corpus
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i]) # Keep all letters 
    review = review.lower() # makes text lowercase
    review = review.split() # split review into a list of elements
    ps = PorterStemmer() # Only keeps root of word
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] # taking all the words in the review that are not in the stop words list
    review = ' '.join(review) # reverses list back into a string 
    corpus.append(review)
    
# Creating the Bag of Words Model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) # Puts each word into separate column
X = cv.fit_transform(corpus).toarray() # Sparse matrix of features (independent variables)
y = dataset.iloc[:,1].values # Create dependent variable vector

# Two Approaches to take 
# 1. Could test each classification model and look at model performance criteria to decide best model
# 2. Most commonly used models for NLP are Naive Bayes, Decision Tree or Random Forest Classification

# Copied from Naive Bayes template 
# Don't need to change anything, only input needed to train machine learning model are x and y
# But could remove scaling because it is not necessary
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Accuracy
print('Accuracy',(55+91)/200)
