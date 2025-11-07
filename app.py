from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os
import pandas as pd

app = Flask(__name__)

# ---------- Load Model and Scaler ----------
try:
    model_path = os.path.join(app.root_path, 'model', 'ridge.pkl')
    scaler_path = os.path.join(app.root_path, 'model', 'scaler.pkl')

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    print("✅ Model and scaler loaded successfully!")

except Exception as e:
    print(f"⚠️ Error loading model or scaler: {e}")
    model = None
    scaler = None

# ---------- Define Input Order ----------
input_features = ["Temperature", "RH", "Ws", "Rain", "FFMC", "DMC", "DC", "ISI", "BUI"]

# ---------- Home Page ----------
@app.route('/')
def home():
    return render_template('index.html')

# ---------- Prediction Route ----------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or scaler is None:
            return jsonify({'prediction': 'Error: Model not loaded'})

        # Collect form data
        data_dict = {}
        for feature in input_features:
            value = request.form.get(feature)
            try:
                data_dict[feature] = float(value)
            except (ValueError, TypeError):
                data_dict[feature] = 0

        # Convert to DataFrame
        input_df = pd.DataFrame([data_dict])
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]

        # Determine risk category
        if prediction < 20:
            risk = "Low Risk"
        elif prediction < 40:
            risk = "Medium Risk"
        else:
            risk = "High Risk"

        result_text = f"🔥 Predicted Fire Risk Index: {prediction:.2f}"
        return jsonify({'prediction': result_text, 'risk_level': risk})

    except Exception as e:
        return jsonify({'prediction': f"Error: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)
