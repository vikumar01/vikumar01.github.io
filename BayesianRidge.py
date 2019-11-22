

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
#dataset = pd.read_csv('181105_missing-data.csv')
dataset = pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv')
dataset['Age']=dataset['Age'].fillna(dataset['Age'].mean())
dataset['Profession']=dataset['Profession'].fillna(method='ffill')
dataset['Year of Record']=dataset['Year of Record'].fillna(dataset['Year of Record'].mean())
dataset['Size of City']=dataset['Size of City'].fillna(dataset['Size of City'].mean())
dataset['Body Height [cm]']=dataset['Body Height [cm]'].fillna(dataset['Body Height [cm]'].mean())

dataset['Country']=dataset['Country'].fillna(method='ffill')
dataset['University Degree']=dataset['University Degree'].fillna(method='ffill')
dataset['Hair Color']=dataset['Hair Color'].fillna(method='ffill')
dataset['Gender']=dataset['Gender'].fillna(method='ffill')
dataset=dataset.dropna(axis=0,how='any')
df = pd.DataFrame(dataset)
#print(df)
#X = dataset.iloc[:, :-1].values #get a copy of dataset exclude last column
X=df[['Year of Record','Age','Country','Size of City','Profession','University Degree']]
X=pd.get_dummies(X, prefix_sep='_')
#y = dataset.iloc[:, 1].values #get array of dataset in column 1st
y= df['Income in EUR']

# =============================================================================
# z=np.abs(stats.zscore(df))
# s=np.where(z>3)
# df=df[(z<3).all(axis=1)]
# df.count()
# =============================================================================

dataset2 = pd.read_csv('tcd ml 2019-20 income prediction test (without labels).csv')
dataset2.isnull().any()
df2 = pd.DataFrame(dataset2)
df2['Age']=df2['Age'].fillna(df2['Age'].mean())
df2['Profession']=df2['Profession'].fillna(method='ffill')
df2['Year of Record']=df2['Year of Record'].fillna(df2['Year of Record'].mean())
df2['Size of City']=df2['Size of City'].fillna(df2['Size of City'].mean())
df2['Body Height [cm]']=df2['Body Height [cm]'].fillna(df2['Body Height [cm]'].mean())
df2['Country']=df2['Country'].fillna(method='ffill')
df2['University Degree']=df2['University Degree'].fillna(method='ffill')
df2['Hair Color']=df2['Hair Color'].fillna(method='ffill')
df2['Gender']=df2['Gender'].fillna(method='ffill')
x1=df2[['Year of Record','Age','Country','Size of City','Profession','University Degree']]
x1=pd.get_dummies(x1, prefix_sep='_')

#y1=df2['Income']


for column in X:
    if column not in x1:
            x1[column]=0

for column in x1:
    if column not in X:
        X[column]=0   

X=X.sort_index(axis=1)
x1=x1.sort_index(axis=1)
from sklearn.model_selection import train_test_split 
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.33, random_state=0)


# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import BayesianRidge
regressor = BayesianRidge()
fitResult = regressor.fit(Xtrain, Ytrain)
YPredTest = regressor.predict(Xtest)

print('Intercept: \n', regressor.intercept_)
print('Coefficients: \n', regressor.coef_)


df2.head()
# Predicting the Test set results
y_pred = regressor.predict(x1)
#print(y_pred)

df2['Income']=y_pred
#print(df2)
df2.describe()

df2.to_csv(r'ml_data_final.csv',index=False)

