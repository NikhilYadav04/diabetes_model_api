from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load the pre-trained model
with open('diab3.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define the mappings
mappings = {
    'Gender': {'Male': 1, 'Female': 0},
    'Age': {'less than 40': 0, '40-49': 1, '50-59': 2, '60 or older': 3},
    'PhysicallyActive': {'one hr or more': 0, 'more than half an hr': 1, 'less than half an hr': 2, 'none': 3},
    'JunkFood': {'occasionally': 0, 'often': 1, 'very often': 2, 'always': 3},
    'BPLevel': {'low': 0, 'normal': 1, 'high': 2},
    'UriationFreq': {'not much': 0, 'quite often': 1},
    'Stress': {'not at all': 0, 'sometimes': 1, 'very often': 2, 'always': 3}
}

# Define the feature order
features = ['Age', 'Gender', 'Family_Diabetes', 'highBP', 'PhysicallyActive', 'BMI',
            'Smoking', 'Alcohol', 'Sleep', 'SoundSleep', 'RegularMedicine',
            'JunkFood', 'Stress', 'BPLevel', 'Pregancies', 'Pdiabetes',
            'UriationFreq']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = []

    for feature in features:
        value = data.get(feature)
        if feature in mappings:
            value = mappings[feature].get(value, value)
        input_data.append(value)

    # Perform the required operations
    input_data[0] /= 10  # Age
    input_data[5] /= 100  # BMI
    input_data[8] /= 10  # Sleep
    input_data[9] /= 10  # SoundSleep
    input_data[12] /= 10  # Stress
    input_data[13] /= 10  # BPLevel
    input_data[4] /= 10  # PhysicallyActive
    input_data[11] /= 10  # JunkFood
    input_data[14] /= 10  # Pregancies

    # Convert to numpy array and reshape for prediction
    input_array = np.array(input_data).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_array)

    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

