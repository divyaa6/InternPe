import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

app=Flask(__name__)

m=pickle.load(open("model.pkl","rb"))

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)
@app.route("/")

def Home():
    return render_template("home.html")

@app.route("/predict",methods=["POST"])
def predict():
    features=[float(x) for x in request.form.values()]
    f=[np.array(features)]
    sc = scaler.transform(f)

    prediction=m.predict(sc)
    output = 'Diabetes' if prediction[0] == 1 else 'No Diabetes'

    return render_template("home.html",prediction_text="Result:{}".format(output))

if __name__=="__main__":
    app.run(debug=True)