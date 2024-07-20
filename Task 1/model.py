import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

df=pd.read_excel('diabetes.xls')
X=df[['Age', 'Glucose', 'BMI', 'Pregnancies']]
y=df.iloc[:,-1].values

scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X_sc,y,test_size=0.2,random_state=42)
model_svm = svm.SVC(kernel='linear')
model_svm.fit(X_train,y_train)


with open("model.pkl", "wb") as model_file:
    pickle.dump(model_svm, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
  pickle.dump(scaler, scaler_file)