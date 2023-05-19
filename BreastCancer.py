#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 19:16:26 2023

@author: ranaemad
"""

#Loading libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

#Load dataset

data = pd.read_csv('data.csv')


#Head
print(data.head())

#Tail
print(data.tail())

#Shape
print(data.shape)

#Information
print(data.info())

#Describtion
print(data.describe())

#Distribution
print(data.groupby('diagnosis').size())

#Another way of distribution
data['diagnosis'].value_counts()

#Checking the columns to see what we need to drop
print(data.columns)

#Drop the non immportant columns
data.drop('id', axis=1, inplace = True)
data.drop('Unnamed: 32', axis=1, inplace = True)

#Checking 
print(data.columns)

#Convert the string values of diagnosis to numerical values
data['diagnosis'].replace({"M" : "1", "B" : "0"}, inplace = True)

#See a peek of the data to make sure of the change
print(data.head())

#Correlation between columns
plt.figure(figsize=(20,20))
sns.heatmap(data.corr(),annot=True,cmap= "mako")

#Machine learning (splitting data)
x = data.drop(['diagnosis'], axis=1)
y = data['diagnosis']

#Splitting the data to testing and training
x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.30,random_state=101)

#Setting standard scaler
s = StandardScaler()

x_train = s.fit_transform(x_train)
x_test = s.fit_transform(x_test)

#Creating Logistic Regression model
logmodel = LogisticRegression()

#Fitting the training data
logmodel.fit(x_train, y_train)

#Predicting on the test data
pred = logmodel.predict(x_test)

#Using confusion matrix without Normalization
ConfusionMatrixDisplay.from_estimator(logmodel, x_test, y_test)
plt.show()

#Normalized confusion matrix
ConfusionMatrixDisplay.from_estimator(logmodel, x_test, y_test,normalize='true')
plt.show()

#Checking the accuracy of logistic regression
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
print(accuracy_score(y_test, pred))

#Creating KNN Classifier model
knn = KNeighborsClassifier(n_neighbors=2) #try k = 2,3,4

#Fitting the training data
knn.fit(x_train,y_train)

#Predicting on the test data
pred = knn.predict(x_test)

#Confusion matrix without Normalization
ConfusionMatrixDisplay.from_estimator(knn, x_test, y_test)
plt.show()

#Normalized confusion matrix
ConfusionMatrixDisplay.from_estimator(knn, x_test, y_test,normalize='true')
plt.show()

#Checking the accuracy of KNeighbors model
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
print(accuracy_score(y_test, pred))


#Creating Random Forset Classifier model
rfc = RandomForestClassifier(n_estimators=100, random_state=0)

#Fitting the training data
rfc.fit(x_train,y_train)

#Predicting on the test data
pred = rfc.predict(x_test)

#Confusion matrix without Normalization
ConfusionMatrixDisplay.from_estimator(rfc, x_test, y_test)
plt.show()

#Normalized confusion matrix
ConfusionMatrixDisplay.from_estimator(rfc, x_test, y_test,normalize='true')
plt.show()

#Checking the accuracy of RFC model
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
print(accuracy_score(y_test, pred))


#Creating Decision Tree Classifier model
dt = DecisionTreeClassifier(random_state=42)

#Fitting the training data
dt.fit(x_train,y_train)

#Predicting on the test data
pred = dt.predict(x_test)

#Confusion matrix without Normalization
ConfusionMatrixDisplay.from_estimator(dt, x_test, y_test)
plt.show()

#Normalized confusion matrix
ConfusionMatrixDisplay.from_estimator(dt, x_test, y_test,normalize='true')
plt.show()

#Checking the accuracy of Decision Tree model
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
print(accuracy_score(y_test, pred))

#Creating GaussianNB model
gnb = GaussianNB()

#Fitting the training data
gnb.fit(x_train,y_train)

#Predicting on the test data
pred = gnb.predict(x_test)

#Confusion matrix without Normalization
ConfusionMatrixDisplay.from_estimator(gnb, x_test, y_test)
plt.show()

#Normalized confusion matrix
ConfusionMatrixDisplay.from_estimator(gnb, x_test, y_test,normalize='true')
plt.show()

#Checking the accuracy of GaussianNB model
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
print(accuracy_score(y_test, pred))
