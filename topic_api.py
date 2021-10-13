#importing the dependencies
import os
import numpy as np
import pandas as pd
import re
import random
import nltk
nltk.download("popular")
nltk.download('stopwords')
# Using the stopwords.
from nltk.corpus import stopwords
# Initialize the stopwords
stoplist = stopwords.words('english')
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from google.colab import files
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
import pickle
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

class Pipeline(object):
  """ data pipeline for data processing and prediction """
  def __init__(self, path_to_pickle='model.pkl'):
    """module intialization    
    PARAMETERS:
    ----------
    Path_to_pickle: path to pre-trained pickle file
    path_to_income: path to the incoming data
    """
    with open(path_to_pickle,'rb') as f:
      self.model = pickle.load(f)

  def _validate(self, json_input):
    """ funtion to validate the input data
    PARAMETERS:
    ----------
    json_input: JSON format
    
    Returns:
    -------
    data:pandas.DataFrame
    """
    df = pd.read_json(json_input)
    columns = ["Text"]
    for col in columns:
      if col not in df.columns:
        df[col] = None
    
    return df[columns] 

  def _preprocess(self, df):
    """ function to preprocess the input data

    Parameters:
    ----------
    data:pandas.DataFrame.It should have the following columns:
    coulmns = ["Text"] 

    Returns:
    -------
    data:pandas.DataFrame
    """

    #PreProcessing
    ##cleaning
    df['Cleaned'] = df['Text'].apply(lambda x: "".join(x.lower() for x in str ((x.split()))))
    df['Cleaned'] = df['Cleaned'].str.replace(r"[^a-zA-Z ]+"," ").replace('\s+', ' ',regex=True)
    stop = stopwords.words("english")
    df.Cleaned = df.Cleaned.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    lemmatizer = WordNetLemmatizer()
    df.Cleaned=df.Cleaned.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    df['Cleaned_lemmatized']=df.Cleaned.apply(lambda x:"".join([lemmatizer.lemmatize(y) for y in x]))  
    return df
    
  def predict(self, json_input):
    """ function to predict
    Parameter
    ---------
    json input:Json format , it expects a list-like JSON input,
    [{column - >value}].

    Returns
    -------
    result: Json format [{input column -> value},{predicted_label - > value},{confidence_score - > value}]

    """

    df = self._validate(json_input)
    df = self._preprocess(df)
    f_content = tf.io.gfile.GFile('tfidf.pkl','rb').read()
    with open('binary_object_stream','wb') as f:
      f.write(f_content)
    tfidftest_sb = joblib.load(open("binary_object_stream","rb"))
    tdidftest = tfidftest_sb.transform(df['Cleaned_lemmatized'])
    predicted_label = self.model.predict(tdidftest)
    confidence_score = self.model.precit_proba(tdidftest)   

    x = {
          "statement": df['Text'],
          "predicted label": y_pred,
          "confidence_score": y_pred_proba
         
        }

    return x 
  




              