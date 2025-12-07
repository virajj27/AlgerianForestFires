from flask import Flask,request,render_template,jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)

app = application

ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/')
def home():
    return "Welcome to the Algerian Forest Fires Prediction App!"

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        try:
            temperature = float(request.form.get('temperature'))
            humidity = float(request.form.get('humidity'))
            wind = float(request.form.get('wind'))
            rain = float(request.form.get('rain'))
            ffmc = float(request.form.get('ffmc'))
            dmc = float(request.form.get('dmc'))
            dc = float(request.form.get('dc'))
            isi = float(request.form.get('isi'))
            bui = float(request.form.get('bui'))

            input_features = np.array([[temperature, humidity, wind, rain, ffmc, dmc, dc, isi, bui]])
            input_features_scaled = scaler.transform(input_features)
            prediction = ridge_model.predict(input_features_scaled)
            output = round(prediction[0], 2)
            return render_template('home.html',result = output)
        except Exception as e:
            return jsonify({'error': str(e)})
    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(host = "0.0.0.0")