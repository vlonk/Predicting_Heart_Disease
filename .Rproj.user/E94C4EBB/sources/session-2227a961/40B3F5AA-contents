import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlip.pyplot as plt

data = pd.read_excel("/Users/ny/COSC 3337/Predicting_Heart_Disease/heart-disease2.xlsx")

def splitdataset(data):
  x = data.values[:, 1:5]
  y = data.values[:, 0]
  
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)
  
  return x, y, x_train, x_test, y_train, y_test

def train_using_gini(x_train, x_test, y_train):
  clf_gini = DecisionTreeClassifier(criterion="gini", random_state = 100, max_depth = 3, min_samples_leaf = 5)
  clf_gini.fit(x_train, y_train)
  return clf_gini

def train_using_entropy(x_train, x_test, y_train):
  clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth = 3, min_samples_leaf = 5)
  clf_entropy.fit(x_train, y_train)
  return clf_entropy

def prediction(x_test, clf_object):
  y_pred = clf_object.predict(x_test)
  print("Predicted values:")
  print(y_pred)
  return y_pred
