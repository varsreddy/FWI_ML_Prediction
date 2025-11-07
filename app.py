from flask import Flask, render_template, request
import numpy as np
import pickle
import os

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

# ---------- Home Page ----------
@app.route('/')
def home():
    return render_template('index.html')

# ---------- Prediction Route ----------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or scaler is None:
            return render_template('index.html', prediction_text="Error: Model not loaded")

        # Collect input safely
        data = []
        for key in request.form:
            try:
                data.append(float(request.form[key]))
            except ValueError:
                data.append(0)  # default if input is missing or invalid

        # Scale and predict
        scaled_input = scaler.transform([data])
        prediction = model.predict(scaled_input)[0]

        result_text = f"🔥 Predicted Fire Risk Index: {prediction:.2f}"
        return render_template('index.html', prediction_text=result_text)

    except Exception as e:
        print(f"⚠️ Prediction Error: {e}")
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
