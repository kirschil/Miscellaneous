# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 08:58:22 2019

@author: kirschil
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

data = pd.read_csv("Mall_Customers.csv")

# Convert Class
data['CustomerID'] = pd.to_numeric(data['CustomerID'])

# Summary of dataset
data.shape
data.describe()
type(data['Age'])

# Basic Plotting
plt.hist(data['Age'], alpha=0.5, facecolor='blue')

by_age = data.groupby('Age').mean()
print(by_age)
by_age2 = by_age['Spending_Score']

plot_by_age = by_age2.plot(title='Spending by Age')
plot_by_age.set_xlabel('Age')
plot_by_age.set_ylabel('Mean Spending Score')

# Row, Column numerically starting at zero
data.iloc[0,0]

# Call Columns by name
data.loc[:, ['Age','Annual_Income']]
data.loc[0:4, 'Age':'Annual_Income']

# Pandas Filtering
age67 = data.loc[data['Age'] == 67, 'CustomerID':'Spending_Score']

# Pandas Select
lesstwocol = data.loc[0:4, ['CustomerID', 'Age','Spending_Score']]
lessonecol = data.drop(columns=['Age'])

# Pandas Mutate
data['Spending_Power'] = (data['Spending_Score'] * data['Annual_Income'])
data = data.assign(State = 'Ohio')

# Pandas Reorder Columns
df2 = data[['CustomerID', 'State', 'Annual_Income']]

# Pandas Arrange
data.sort_values(by=['Age', 'Annual_Income'])

