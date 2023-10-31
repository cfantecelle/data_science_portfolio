#######################################################
## Creating app to check property values in SP       ##
#######################################################

# Importing packages
import numpy as np
from flask import Flask, jsonify, request
from flask_restful import Resource, Api
from joblib import load

##

# Creating flask object
app = Flask(__name__)

# API
api = Api(app)

# Loading model
model = load("data/model.joblib")


class PropertyValue(Resource):
    def get(self):
        return("Property Value Prediction")
    
    def post(self):
        args = request.get_json(force=True)
        input_values = np.asarray(list(args.values())).reshape(1, -1)
        predict = model.predict(input_values)[0]

        return jsonify({'Property Value Prediction: ': float(predict)})
    

api.add_resource(PropetyValue, '/')


if __name__ == '__main__':
    app.run()
    