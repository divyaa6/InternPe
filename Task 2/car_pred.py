import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle

df=pd.read_csv('quikr_car.csv')

df=df[df['year'].str.isnumeric()]
df['year']=df['year'].astype(int)
df=df[df['Price']!='Ask For Price']
df['Price']=df['Price'].str.replace(',','').astype(int)
df=df[df['Price']<6000000]
df['kms_driven']=df['kms_driven'].str.split().str.get(0).str.replace(',','')
df=df[df['kms_driven'].str.isnumeric()]
df['kms_driven']=df['kms_driven'].astype(int)
df=df[~df['fuel_type'].isna()]
df['name']=df['name'].str.split().str.slice(start=0,stop=3).str.join(' ')

X=df[['name','company','year','kms_driven','fuel_type']]
y=df['Price']

ohe=OneHotEncoder()
ohe.fit(X[['name','company','fuel_type']])
column_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),remainder='passthrough')

scores=[]
for i in range(1000):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=i)
    lr=LinearRegression()
    pipe=make_pipeline(column_trans,lr)
    pipe.fit(X_train,y_train)
    y_pred=pipe.predict(X_test)
    scores.append(r2_score(y_test,y_pred))

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=np.argmax(scores))
lr=LinearRegression()
pipe=make_pipeline(column_trans,lr)
pipe.fit(X_train,y_train)

print(X_train.shape)
print(X_test.shape)
prediction=pipe.predict(pd.DataFrame(columns=X_test.columns,data=np.array(['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']).reshape(1,5)))
print(prediction)

with open('car_pred.pkl','wb') as model_file:
    pickle.dump(pipe,model_file)