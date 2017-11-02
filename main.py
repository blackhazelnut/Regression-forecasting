import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

df = quandl.get('WIKI/GOOGL')
print(df.head)
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]
df['HL_PCT'] = (df['Adj. High']-df['Adj. Close'])/df['Adj. Close']*100.0
df['PCT_change'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100.0

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forecast_col='Adj. Close'

df.fillna(-99999, inplace=True)#fill nan with value
#if remove the whole row will end out losing information
#it will treated as outlier 

forecast_out = int(math.ceil(0.01*len(df))) #predict out 10% of the data
#print(forecast_out) 34
df['label']=df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

X = np.array(df.drop(['label'],1))
y = np.array(df['label'])
X = preprocessing.scale(X)
y=np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

#clf = LinearRegression()
clf = svm.SVR()
clf.fit(X_train,y_train)
print(X_train)
# print(X_train.shape)
#print(y_train.shape)
acc = clf.score(X_test,y_test)
#print(X_test.shape)
#print(y_test.shape)
print(acc)
