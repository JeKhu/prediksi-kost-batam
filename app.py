from flask import Flask, render_template, redirect, url_for, request, jsonify
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle
import joblib

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
ENCODER_MAPPING = pickle.load(open('encoder_mapping.pkl', 'rb'))

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    colnames = ['kecamatan','ukuran','air_listrik','ac','air_panas','wc_dalam','kasur','kipas_angin','kloset_duduk','kursi','lemari','meja','tv','tidak_ada','dapur','wc','kulkas','laundry','mesin_cuci','parkir_mobil','parkir_motor','tv_luar','ruang_santai','wifi']
    info_for_prediction = []
    for key in colnames:
        if request.form.get(key, None):
            info_for_prediction.append(ENCODER_MAPPING[key]['Yes' if request.form[key] == 'check' else request.form[key]])
        else:
            info_for_prediction.append(ENCODER_MAPPING[key]['No'])
    prediction = model.predict([info_for_prediction])
    return jsonify({ 'harga': ENCODER_MAPPING['harga'][prediction[0]] })
    

if __name__ == '__main__':
    app.run(port=3000, debug=True)