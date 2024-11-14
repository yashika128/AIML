from flask import Flask, request, render_template_string
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

# Updated HTML form with enhanced styling
form_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Diabetes Prediction Form</title>
    <style>
        /* Styling for the page */
        body {
            background-color: #f0f8ff;
            font-family: Arial, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }
        .container {
            background-color: #ffffff;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0px 8px 20px rgba(0, 0, 0, 0.15);
            width: 100%;
            max-width: 450px;
            box-sizing: border-box;
            text-align: center;
        }
        h2 {
            color: #333;
            margin-bottom: 20px;
        }
        label {
            display: block;
            font-size: 16px;
            color: #333;
            margin-bottom: 6px;
            text-align: left;
        }
        input[type="number"] {
            width: 100%;
            padding: 12px;
            margin-bottom: 16px;
            border: 1px solid #ccc;
            border-radius: 6px;
            font-size: 16px;
        }
        input[type="submit"] {
            width: 100%;
            padding: 12px;
            font-size: 16px;
            color: white;
            background-color: #28a745;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        input[type="submit"]:hover {
            background-color: #218838;
        }
        .prediction-text {
            font-size: 18px;
            color: #555;
            margin-top: 20px;
            font-weight: bold;
        }
    </style>
</head>
<body>

    <div class="container">
        <h2>Diabetes Prediction</h2>
        <form method="POST" action="/predict">
            <label for="pregnancies">Pregnancies:</label>
            <input type="number" step="any" name="pregnancies" required>

            <label for="glucose">Glucose Level:</label>
            <input type="number" step="any" name="glucose" required>

            <label for="blood_pressure">Blood Pressure:</label>
            <input type="number" step="any" name="blood_pressure" required>

            <label for="skin_thickness">Skin Thickness:</label>
            <input type="number" step="any" name="skin_thickness" required>

            <label for="insulin">Insulin Level:</label>
            <input type="number" step="any" name="insulin" required>

            <label for="bmi">BMI:</label>
            <input type="number" step="any" name="bmi" required>

            <label for="dpf">Diabetes Pedigree Function:</label>
            <input type="number" step="any" name="dpf" required>

            <label for="age">Age:</label>
            <input type="number" step="any" name="age" required>

            <input type="submit" value="Predict">
        </form>

        {% if prediction_text %}
            <div class="prediction-text">
                <p>{{ prediction_text }}</p>
            </div>
        {% endif %}
    </div>

</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(form_html)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect the input features from the form
        features = [
            float(request.form['pregnancies']),
            float(request.form['glucose']),
            float(request.form['blood_pressure']),
            float(request.form['skin_thickness']),
            float(request.form['insulin']),
            float(request.form['bmi']),
            float(request.form['dpf']),
            float(request.form['age'])
        ]

        # Scale the input features
        features_scaled = scaler.transform([features])

        # Make a prediction using the loaded model
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0][1]

        # Translate prediction into a readable format
        result = "Diabetes Detected" if prediction == 1 else "No Diabetes Detected"
        probability = f"Probability of Diabetes: {prediction_proba:.2f}"

        return render_template_string(form_html, prediction_text=f"{result}. {probability}")

    except Exception as e:
        print(f"Error: {str(e)}")
        return render_template_string(form_html, prediction_text="Error: Invalid input or internal error.")

if __name__ == '__main__':
    app.run(debug=True)
