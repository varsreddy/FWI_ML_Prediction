from flask import Flask, render_template, request, send_from_directory
import numpy as np
import pickle
import os

app = Flask(__name__, template_folder='templates')

# Load model and scaler
with open(os.path.join('Results-plots,pkl', 'ridge.pkl'), 'rb') as f:
    model = pickle.load(f)

with open(os.path.join('Results-plots,pkl', 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

# ✅ Route to serve CSS from templates folder
@app.route('/templates/<path:filename>')
def serve_template_files(filename):
    return send_from_directory(os.path.join(app.root_path, 'templates'), filename)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(x) for x in request.form.values()]
        scaled_input = scaler.transform([data])
        prediction = model.predict(scaled_input)[0]
        result_text = f"🔥 Predicted Fire Risk Index: {prediction:.2f}"
        return render_template('index.html', prediction_text=result_text)
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
