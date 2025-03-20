from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
try:
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Debugging: Print received JSON data
        data = request.get_json()
        print("Received Data:", data)

        # Extracting input values
        lot_area = float(data['features'][0])
        year_built = int(data['features'][1])
        total_rooms = int(data['features'][2])
        garage_area = float(data['features'][3])

        # Debugging: Print extracted values
        print(f"Features: {lot_area}, {year_built}, {total_rooms}, {garage_area}")

        # Creating a feature array
        features = np.array([[lot_area, year_built, total_rooms, garage_area]])

        # Making a prediction
        prediction = model.predict(features)

        # Debugging: Print prediction
        print(f"Prediction: {prediction[0]}")

        return jsonify({'predicted_price': float(prediction[0])})
    
    except Exception as e:
        print("Error:", e)  # Print error in console for debugging
        return jsonify({'error': 'Error predicting price. Please try again.'}), 400

if __name__ == "__main__":
    app.run(debug=True)
