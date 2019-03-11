# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 09:48:41 2019

@author: kirschil
"""

import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
#from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



os.chdir('C:\\Users\\kirschil\\Downloads')

data = pd.read_csv("msaboost.csv")

cols = [0, 1, 2, 3, 5, 40, 41]
data.drop(data.columns[cols],axis=1,inplace=True)

# MSA Nowcasting

train_label = data.MSA_Recession[0:12000]
test_label = data.MSA_Recession[12001:22042]

train_label.describe()
test_label.describe()

train_label = train_label.astype('bool')
test_label = test_label.astype('bool')

data.drop(data.columns[1],axis=1,inplace=True)


train = data[0:12000]
#x = train.values 
#min_max_scaler = preprocessing.MinMaxScaler()
#x_scaled = min_max_scaler.fit_transform(x)
#train = pd.DataFrame(x_scaled)

test = data[12001:22042]
#x = test.values 
#min_max_scaler = preprocessing.MinMaxScaler()
#x_scaled = min_max_scaler.fit_transform(x)
#test = pd.DataFrame(x_scaled)


logreg = LogisticRegression()
logreg.fit(train, train_label)
y_pred=logreg.predict(test)


cnf_matrix = metrics.confusion_matrix(test_label, y_pred)
cnf_matrix

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

print("Accuracy:",metrics.accuracy_score(test_label, y_pred))

# Nowhere near XGBOOST in R (as expected)

from xgboost import XGBClassifier


model = XGBClassifier()
model.fit(train, train_label)




y_pred = model.predict(test)

print("Accuracy:",metrics.accuracy_score(test_label, y_pred))

cnf_matrix2 = metrics.confusion_matrix(test_label, y_pred)
cnf_matrix2

