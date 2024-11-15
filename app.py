from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model (make sure 'oil_quality_model.pkl' exists)
model = pickle.load(open('oil_quality_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get temperature, capacity, and color from form
        try:
            temperature = float(request.form['temperature'])
            capacity = float(request.form['capacity'])
            color = float(request.form['color'])
        except ValueError:
            return render_template('index.html', prediction_text="Invalid input! Please enter valid numbers.")
        
        # Prepare input data for model prediction
        features = np.array([[temperature, capacity, color]])
        
        # Predict oil quality
        prediction = model.predict(features)[0]
        
        # Map prediction to quality output
        quality = "Good" if prediction == 1 else "Bad"
        return render_template('index.html', prediction_text=f'The oil quality is predicted to be: {quality}')
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
