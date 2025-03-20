import os
import pickle
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Dynamically get the correct path to the model file
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")

# Load the model
with open(model_path, "rb") as f:
    model = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({"predicted_price": prediction[0]})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
