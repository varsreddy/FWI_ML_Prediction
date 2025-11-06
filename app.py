from flask import Flask, render_template, request, send_from_directory
import numpy as np
import pickle
import os

app = Flask(__name__, template_folder='templates')

# ---------- Load Model and Scaler ----------
try:
    model_path = os.path.join(app.root_path, 'Results-plots,pkl', 'ridge.pkl')
    scaler_path = os.path.join(app.root_path, 'Results-plots,pkl', 'scaler.pkl')

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

except Exception as e:
    print(f"⚠️ Error loading model or scaler: {e}")


# ---------- Serve CSS from templates ----------
@app.route('/style.css')
def serve_css():
    return send_from_directory('templates', 'style.css')


# ---------- Home Page ----------
@app.route('/')
def home():
    return render_template('index.html')


# ---------- Prediction Route ----------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(x) for x in request.form.values()]
        scaled_input = scaler.transform([data])
        prediction = model.predict(scaled_input)[0]

        result_text = f"🔥 Predicted Fire Risk Index: {prediction:.2f}"
        return render_template('index.html', prediction_text=result_text)

    except Exception as e:
        print(f"⚠️ Prediction Error: {e}")
        return render_template('index.html', prediction_text=f"Error: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)
