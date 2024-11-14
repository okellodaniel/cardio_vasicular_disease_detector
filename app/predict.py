import os
import pickle
import logging

import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from flask import Flask, jsonify, request

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

prediction_model_path = '../model/model.bin'

model = None
dv = None

try:
    dv, model = pickle.load(open(prediction_model_path, 'rb'))
    logger.info("DictVectorizer instance and model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load the model: {e}")
    dv = None
    model = None


@app.route('/', methods=['POST'])
def predict():
    try:
        patient = request.get_json()
        logger.info(f"Received patient data: {patient}")

        prediction = predict_cardio(patient, model, dv)
        logger.info(f"Prediction: {prediction}")

        prediction_rounded = round(float(prediction), 4)

        return jsonify(
            {
                'prediction': prediction_rounded,
                'cardio': bool(prediction >= 0.5),
                'predictionPercent': f"{prediction_rounded*100}%",
                'threshold': float(0.5)
            })

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({'error': 'Prediction failed'}), 500


def predict_cardio(patient, model, dv):
    logger.debug(f"Creating DMatrix for {patient} object")

    try:
        pratient_transformed = dv.transform(patient)

        patient_dmatrix = xgb.DMatrix(pratient_transformed)

        y_pred = model.predict(patient_dmatrix)

        return y_pred[0]

    except Exception as e:
        logger.error("Error: %s" % e)


if __name__ == '__main__':
    app.run(debug=True)
