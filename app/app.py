# Run me using "flask run" while within the app directory in cli.
# send post requests to "http://127.0.0.1:5000/usa-rain-predictor/"

# URLs to useful resources...
# RESTful APIs for models -> https://medium.com/@einsteinmunachiso/rest-api-implementation-in-python-for-model-deployment-flask-and-fastapi-e80a6cedff86
# Serializing/Deserializing models -> https://medium.com/@einsteinmunachiso/saving-your-machine-learning-model-in-python-pickle-dump-b01ae60a791c

import numpy as np
import pickle # for deserialization of saved model
from flask import Flask, request, jsonify 

import warnings
warnings.filterwarnings("ignore")

# Creating a flask instance.
app = Flask(__name__)

# Model and encoder paths.
model_file_path = "../model.pkl"
location_encoder_path = "../location_encoder.pkl"

# deserializing saved model and encoder.
with open(model_file_path, "rb") as file:
    model = pickle.load(file)

with open(location_encoder_path, "rb") as file:
    loc_enc = pickle.load(file)


@app.post("/usa-rain-predictor/")
def usa_rain_predictor_function():
    json_data = request.json # Getting everything posted to the endpoint

    # Extracting all the info we need from the endpoint to make predictions.
    location = json_data.get("location")
    temperature = json_data.get("temperature")
    humidity = json_data.get("humidity")
    wind_speed = json_data.get("wind_speed")
    precipitation = json_data.get("precipitation")
    cloud_cover = json_data.get("cloud_cover")
    pressure = json_data.get("pressure")

    # feature engineering (encoding location from text to int as we did using the exact same encoder used during training).
    location = loc_enc.transform([location]) # using transform only because we do not need to re-fit.
    # print(location)

    # Making prediction
    features = [location[0], temperature, humidity, wind_speed, precipitation, cloud_cover, pressure] # organizing information in the right order (order used during training)
    result = model.predict([features])
    # print(result)

    json_response = {
        "Prediction": str(result[0]),
    }

    return jsonify(json_response)