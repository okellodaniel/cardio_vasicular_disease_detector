import os
import pickle
import logging
import uvicorn

import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from fastapi.encoders import jsonable_encoder
from scalar_fastapi import get_scalar_api_reference

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Load model and DictVectorizer
prediction_model_path = 'model.bin'

model = None
dv = None

try:
    dv, model = pickle.load(open(prediction_model_path, 'rb'))
    logger.info("DictVectorizer instance and model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load the model: {e}")
    dv = None
    model = None

# schema


class PatientData(BaseModel):
    age: int = Field(..., description="Age of the patient in days")
    gender: Literal["male",
                    "female"] = Field(..., description="Gender of the patient")
    height: float = Field(...,
                          description="Height of the patient in centimeters")
    weight: float = Field(...,
                          description="Weight of the patient in kilograms")
    ap_hi: float = Field(..., description="Systolic blood pressure")
    ap_lo: float = Field(..., description="Diastolic blood pressure")
    cholesterol: Literal["normal", "above_normal", "well_above_normal"] = Field(..., description="Cholesterol level"
                                                                                )
    gluc: Literal["normal", "above_normal", "well_above_normal"] = Field(..., description="Glucose level"
                                                                         )
    smoke: Literal["yes", "no"] = Field(..., description="Smoking status")
    alco: Literal["yes", "no"] = Field(...,
                                       description="Alcohol consumption status")
    active: Literal["yes",
                    "no"] = Field(..., description="Physical activity status")


def predict_cardio(patient, model, dv):
    logger.debug(f"Creating DMatrix for {patient} object")
    try:
        patient_transformed = dv.transform(patient)
        patient_dmatrix = xgb.DMatrix(patient_transformed)
        y_pred = model.predict(patient_dmatrix)
        return y_pred[0]
    except Exception as e:
        logger.error(f"Error during transformation or prediction: {e}")
        raise


@app.post("/")
async def predict(patient: PatientData):
    try:
        patient_data_json = jsonable_encoder(patient)

        logger.info(f"Received patient data: {patient_data_json}")

        prediction = predict_cardio(patient_data_json, model, dv)
        logger.info(f"Prediction: {prediction}")

        prediction_rounded = round(float(prediction), 4)
        return {
            "prediction": prediction_rounded,
            "cardio": bool(prediction >= 0.5),
            "prediction%": f"{prediction_rounded * 100}%",
            "threshold": 0.5
        }
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.get("/", include_in_schema=False)
async def scalar_html():
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title="cardio vascular disease detector",

    )
