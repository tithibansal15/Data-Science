#!/usr/bin/env python
# coding: utf-8

# In[4]:


# First up, I'll import every library that will be used in this project is imported at the start.

# Data handling and processing
import pandas as pd
import numpy as np

# Data visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Statistics
from scipy import stats
import statsmodels.api as sm
from scipy.stats import randint as sp_randint
from time import time

# NLP
import nltk
nltk.download('wordnet')
import re
from textblob import TextBlob
from nltk.corpus import stopwords
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report


# In[15]:


# Reading in data
data = pd.read_excel('Key Activity data.xlsx','Combined')


# In[17]:


data.columns


# ## Data Preparation

# In[35]:


from nltk.stem import WordNetLemmatizer
import re
# Importing SKLearn's list of stopwords and then appending with my own words 
stop = text.ENGLISH_STOP_WORDS

documents = []
stemmer = WordNetLemmatizer()
for sen in range(0, len(data)):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(data['Legacy Activity Description'].iloc[sen]))
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    # Converting to Lowercase
    document = document.lower()
    # Remove Stopwords
    document = ' '.join([word for word in document.split() if word not in (stop)])
    # Remove numbers
    document = re.sub("\d+", "", document)
    # Remove single or double text
    document = " ".join([i for i in document.split(' ') if len(i)>2])
    # Lemmatization
    document = document.split()
    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    documents.append(document)


# In[36]:


data['Key Activity Type'] = [x.lower().strip() for x in data['Key Activity Type']]
data  = data.drop_duplicates()
data['Legacy Activity Description'] = documents


# ## Splitting into Test and Train

# In[46]:


from sklearn.model_selection import train_test_split
X_train_data, X_test_data, y_train, y_test = train_test_split(data['Legacy Activity Description'], data['Key Activity Type'], test_size=0.2, random_state=0)


# In[47]:


from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

vectorizer = CountVectorizer(max_features=1500, stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(documents).toarray()
X


# In[48]:


from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

tfidfconverter = TfidfVectorizer(max_features=1500, stop_words=stopwords.words('english'))
tfidfconverter.fit(documents)
X_train = tfidfconverter.transform(X_train_data)
X_test = tfidfconverter.transform(X_test_data)
tfidfconverter.fit_transform(documents).toarray()


# # Random Forest Classifier

# In[49]:


from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train) 
y_pred = classifier.predict(X_test)


# In[50]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))


# ### Data Post-Processing

# In[62]:


idx = -10
test_text = X_test_data.iloc[idx]
print('Test Legacy Description: ' + test_text)

documents = []
stemmer = WordNetLemmatizer()
document = re.sub(r'\W', ' ', str(test_text))
document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
document = re.sub(r'\s+', ' ', document, flags=re.I)
document = re.sub(r'^b\s+', '', document)
document = document.lower()
document = ' '.join([word for word in document.split() if word not in (stop)])
document = re.sub("\d+", "", document)
document = " ".join([i for i in document.split(' ') if len(i)>2])
document = document.split()
document = [stemmer.lemmatize(word) for word in document]
document = ' '.join(document)

documents.append(document)

X = tfidfconverter.transform(documents)
print('Predicted Class: '+ classifier.predict(X)[0])
print('Actual Class: '+ y_test.iloc[idx])


# In[ ]:




