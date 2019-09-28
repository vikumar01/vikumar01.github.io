

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
#dataset = pd.read_csv('181105_missing-data.csv')
dataset = pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv')
#dataset.isnull().any()
#dataset=dataset.fillna(method='ffill')
dataset=dataset.dropna(axis=0,how='any')
df = pd.DataFrame(dataset)
#print(df)
#X = dataset.iloc[:, :-1].values #get a copy of dataset exclude last column
from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
df['Gender'] = lb_make.fit_transform(df['Gender'])
df['Country'] = lb_make.fit_transform(df['Country'])
df['University Degree'] = lb_make.fit_transform(df['University Degree'])
df['Hair Color'] = lb_make.fit_transform(df['Hair Color'])
X=df[['Year of Record','Gender','Age','Country','Size of City','University Degree','Wears Glasses','Hair Color','Body Height [cm]']]
#y = dataset.iloc[:, 1].values #get array of dataset in column 1st
y= df['Income in EUR']


df.head()
#dummy=pd.get_dummies(df.Country)
#df.join(dummy)
#print(df)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)

print('Intercept: \n', regressor.intercept_)
print('Coefficients: \n', regressor.coef_)


dataset2 = pd.read_csv('tcd ml 2019-20 income prediction test (without labels).csv')
dataset2.isnull().any()
dataset2=dataset2.fillna(method='ffill')
df2 = pd.DataFrame(dataset2)
lb_make = LabelEncoder()
df2['Gender'] = lb_make.fit_transform(df2['Gender'])
df2['Country'] = lb_make.fit_transform(df2['Country'])
df2['University Degree'] = lb_make.fit_transform(df2['University Degree'])
df2['Hair Color'] = lb_make.fit_transform(df2['Hair Color'])

x1=df2[['Year of Record','Gender','Age','Country','Size of City','University Degree','Wears Glasses','Hair Color','Body Height [cm]']]
#y1=df2['Income']

df2.head()
# Predicting the Test set results
y_pred = regressor.predict(x1)
#print(y_pred)

df2['Income']=y_pred
#print(df2)
df2.describe()

df2.to_csv(r'ml_data_final.csv')


