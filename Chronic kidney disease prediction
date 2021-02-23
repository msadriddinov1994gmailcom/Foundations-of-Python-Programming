import pandas as pd
import numpy as np

data = pd.read_csv('Data/kidney_disease.csv')
print(data.shape)
data.head()

data.drop('id', axis=1, inplace=True)
data[['htn', 'ane', 'pe', 'cad', 'dm']] = data[['htn', 'ane', 'pe', 'cad', 'dm']].replace(to_replace={'yes':1, 'no':0})
data['classification'] = data['classification'] == 'ckd'
data[['rbc', 'pc']] = data[['rbc', 'pc']].replace(to_replace={'normal':0, 'abnormal':1})
data[['pcc', 'ba']] = data[['pcc', 'ba']].replace(to_replace={'present':0, 'notpresent':1})
data['appet'] = data['appet'].replace(to_replace={'good':0, 'poor':1})
data.pcv = pd.to_numeric(data.pcv, errors='coerce')
data.wc = pd.to_numeric(data.wc, errors='coerce')
data.rc = pd.to_numeric(data.rc, errors='coerce')
data.dm = pd.to_numeric(data.dm, errors='coerce')
data.cad = pd.to_numeric(data.cad, errors='coerce')
df = data.dropna(axis=0)
print(df.shape)
df.head()

X = df.drop(['classification', 'sg', 'appet', 'rc', 'pcv', 'hemo', 'sod'], axis=1).values
y = df['classification'].values

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

model = LogisticRegression()
knn = KNeighborsClassifier(n_neighbors=5)
k = KFold(n_splits=5, shuffle=True)

splits = k.split(X)
lscores = []
ladd = {}
nscores = []
nadd = {}


for train_incide, test_incide in splits:
    
    X_train, X_test = X[train_incide], X[test_incide]
    y_train, y_test = y[train_incide], y[test_incide]
    
    model.fit(X_train, y_train)
    y_predict1 = model.predict(X_test)
        
    lscores.append(model.score(X_test, y_test))
    
    ladd['Accuracy'] = accuracy_score(y_test, y_predict1)
    ladd['Precision'] = precision_score(y_test, y_predict1)
    ladd['Recall'] = recall_score(y_test, y_predict1)
    ladd['F1'] = f1_score(y_test, y_predict1)
    
    knn.fit(X_train, y_train)
    nscores = (knn.score(X_test, y_test))
    y_predict2 = knn.predict(X_test)

    nadd['Accuracy'] = accuracy_score(y_test, y_predict2)
    nadd['Precision'] = precision_score(y_test, y_predict2)
    nadd['Recall'] = recall_score(y_test, y_predict2)
    nadd['F1'] = f1_score(y_test, y_predict2)
    
    

    

print('Mean of Logisticregression: ', np.mean(lscores))
print(ladd)
print('\nMean of KNeighbors: ', np.mean(nscores))
print(nadd)
