import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model_knn.pkl','rb'))
scaler = pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    final_features = scaler.transform(final_features)
    prediction = model.predict(final_features)
    
    output = prediction
    
    if output == [0]:
        output = "Parkinsons Disease Not Detected"
    elif output == [1]:
        output = "Parkinsons Disease Detected"
    
    return render_template('index.html', prediction_text='Diagnosis Result: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)