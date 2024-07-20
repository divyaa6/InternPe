from flask import Flask,render_template,request
import numpy as np
import pandas as pd
import pickle

app=Flask(__name__)

with open('car_pred.pkl','rb') as model_file:
    model=pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])

def predict():
    
    car_model=request.form.get('f1')
    company=request.form.get('f2')
    year=request.form.get('f3')
    driven=request.form.get('f4')
    fuel_type=request.form.get('f5')
    

    prediction=model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                                          data=(np.array([car_model,company,year,driven,fuel_type]).reshape(1,5))))
    rounded_amount = round(prediction[0], 2)

    return render_template('index.html',prediction_text=f'Price of the car is {rounded_amount}')

if __name__ == '__main__':
    app.run(debug=True)