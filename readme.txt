pyspark_PP.py
-------------

  reddit data preprocessing.

  run on gateway.sfucloud.ca with: spark-submit pyspark_PP.py /courses/318/reddit-2 output

  Requires:

  import sys
  from pyspark.sql import SparkSession, functions, types
  import string, re
  from pyspark.sql.window import Window
  from pyspark.sql.functions import col, rank, desc, udf, to_json
  from pyspark.ml.feature import StopWordsRemover
  from pyspark.ml.feature import RegexTokenizer
  from pyspark.sql.types import IntegerType


NLP2.0.ipynb
------------

  LDA based topic encoding for comments.

  input filename can be changed on [336]. use the two output*.csv as inputs.

  Requires:

  import numpy as np
  import re
  import lda
  import lda.datasets
  import pandas as pd
  from scipy.sparse import coo_matrix
  import math as mt
  from nltk.corpus import stopwords

CLASSIFIER1.0.ipynb

  Runs classfiers on LDA output.

  input file name can be changed on [4]. use the three results*.csv as inputs.

  Requires:

  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt
  import sys
  from sklearn.model_selection import train_test_split
  from sklearn.naive_bayes import GaussianNB
  from sklearn.metrics import accuracy_score
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.svm import SVC
  from sklearn.preprocessing import OneHotEncoder
  from sklearn.metrics import confusion_matrix
  import seaborn as sn
