# Cardio Vascular Disease Detector

## Introduction

Cardiovascular disease (CVD) is the leading cause of mortality globally, often progressing unnoticed until critical stages. Early detection enables timely intervention, saving lives, and reducing healthcare costs. This project leverages machine learning and FastAPI to develop an application for predicting the risk of cardiovascular disease based on individual health indicators.

---
## Problem Description
Cardiovascular diseases are responsible for the highest mortality rate globally. Identifying high-risk individuals early is critical for timely interventions that can save lives and reduce healthcare costs. This project uses the Kaggle Cardiovascular Disease dataset, which contains 70,000 patient records with 13 features, to build a machine learning model capable of predicting the likelihood of cardiovascular disease.

Key benefits of this project include:

- Personalized Risk Assessment: Supports healthcare providers in tailoring interventions to individual needs.
- Proactive Healthcare: Enables earlier treatment, reducing disease progression and associated costs.
- Scalable Deployment: Designed for real-time prediction and integration with healthcare systems.

___

## Architecture Overview

The project is designed for modularity, and efficiency. Below is an architectural overview:

### Components
1. Data Pipeline:
   - Data cleaning, preprocessing, and feature engineering are handled programmatically using a python script.
   - Dataset: Kaggle Cardiovascular Dataset containing 70,000 records.

2. Modeling:
   - XGBoost model for prediction, trained with hyperparameter tuning.
   - `DictVectorizer` for encoding categorical variables.

3. Application Backend:
   - FastAPI Framework: Hosts the prediction API.
   - Logging: Tracks requests and debugging information.
   - Input Validation: Ensured using Pydantic models.

4. Containerization:
   - Docker container with FastAPI and model preloaded for consistent deployment across environments.

5. Deployment:
   - Hosted on [**Fly.io** ](https://fly.io), exposing the API to public requests.

### Data Flow
1. User Input: JSON payload with patient health data is sent to the application prediction endpoint.
2. Data Validation: Pydantic validates and parses input data.
3. Prediction Logic:
   - Data transformed using `DictVectorizer`.
   - Transformed data passed to the XGBoost model.
   - Model outputs prediction probability and binary classification.
4. API Response: JSON response returned to the user.

---

### Project Structure
The project is organized as follows:

1. Notebooks:
   - [`notebook.ipynb`](./notebooks/notebook.ipynb): Contains data preparation, model training, evaluation, and selection processes.
2. Scripts:
   - [`model_trainer.py`](./scripts/model_trainer.py): Encapsulates the data preparation and final model training logic.
3. [`main.py`](./main.py): The FastAPI application script that serves the model predictions.
4. Configuration Files:
   - `Pipfile`: Manages project dependencies.
   - `fly.toml`: Specifies the hosting environment requirements for deployment on fly.io.
5. Dockerfile: Defines the steps to containerize the application using Docker.
6. Dataset:
   - [`cardio_vascular_disease_dataset.csv`](./data/cardio_vascular_disease_dataset.csv): The dataset used for training and evaluation, with its path specified in the .env file as DATASET_PATH.
7. Models:
   - `model.bin`: The serialized model and DictVectorizer saved after training.

___

## Dataset Details

The **Cardiovascular Disease Dataset** was sourced from **[Kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)**. It contains 70,000 patient records with multiple health indicators relevant to predicting cardiovascular disease (CVD). The dataset is intended for binary classification, where the target variable indicates the presence or absence of cardiovascular disease.

#### Dataset Structure

- **Rows:** 70,000
- **Columns:** 12 (11 features + 1 target variable)

---

### Fields Description

| **Field**       | **Description**                                                                                                   | **Type**     |
|------------------|-------------------------------------------------------------------------------------------------------------------|--------------|
| **Age**          | Age of the patient in days.                                                                                      | Numeric      |
| **Gender**       | Gender of the patient (`1` for female, `2` for male).                                                            | Categorical  |
| **Height**       | Height of the patient in centimeters.                                                                            | Numeric      |
| **Weight**       | Weight of the patient in kilograms.                                                                              | Numeric      |
| **AP_hi**        | Systolic blood pressure (higher number in the blood pressure reading).                                           | Numeric      |
| **AP_lo**        | Diastolic blood pressure (lower number in the blood pressure reading).                                           | Numeric      |
| **Cholesterol**  | Cholesterol level (`1`: normal, `2`: above normal, `3`: well above normal).                                      | Categorical  |
| **Gluc**         | Glucose level (`1`: normal, `2`: above normal, `3`: well above normal).                                          | Categorical  |
| **Smoke**        | Indicates whether the patient smokes (`0`: no, `1`: yes).                                                       | Binary       |
| **Alco**         | Indicates whether the patient consumes alcohol (`0`: no, `1`: yes).                                             | Binary       |
| **Active**       | Indicates whether the patient engages in physical activity (`0`: no, `1`: yes).                                 | Binary       |
 **Cardio**       | Target variable indicating the presence of cardiovascular disease (`0`: no, `1`: yes).                          | Binary       |



### Preprocessing Adjustments
   - There were no missing values in the dataset.
   - Features were scaled (e.g., standardization for numerical variables).
   - Outliers in `AP_hi` and `AP_lo` were addressed based on medical thresholds.

This dataset was chosen for its comprehensiveness and relevance to building a machine learning model for cardiovascular disease risk prediction. It includes a mix of categorical, binary, and numerical features that align with real-world patient profiles.


## Model Selection

### Numerical variable correlation

Following EDA on the dataset, a correlation matrix was evaluated and results were aas follows
![alt text](./images/image.png)

Height and Weight (0.29):
There is a moderate positive correlation (0.29) between height and weight, indicating that taller individuals tend to have higher weights. This is the strongest correlation observed in this matrix.
Other Variables and Weak Correlations:
The rest of the correlations between variables are quite low, suggesting minimal linear relationships between these features. For example, age shows weak correlations with all other variables, with the highest correlation being only 0.054 with weight. This means age does not have a strong linear relationship with other variables in this dataset.
Blood Pressure Variables (ap_hi and ap_lo) also show low correlations with each other (0.016), which is unusual, as systolic and diastolic blood pressures are often expected to be correlated. This low correlation might suggest inconsistencies in data recording or could indicate that these variables are independent in this specific dataset.
No Strong Predictors:
The lack of strong correlations suggests that none of these variables alone has a significant linear relationship with others. This could imply that these features may contribute only weakly to each other in terms of prediction if using linear models.
Note: Given the weak relationships, this dataset might benefit from more complex, non-linear analysis techniques, as linear models may not capture significant patterns effectively.

Therefore the choice was to test

Decision tree
Random forest
XGBoost
The best performing model will be based on the best prediction performance.

The project evaluated several models before finalizing **XGBoost**. Below is a comparison of their performance:

### Final Model Comparison

Below is the comparison of models based on their performance metrics:

| Model                 | AUC-ROC  | Accuracy |
|-----------------------|----------|----------|
| **Decision Tree**     | 79.59   | 73.21%   |
| **Random Forest**     | 80.35   | 73.64%   |
| **XGBoost**           | 80.41  | 73.77%   |

### Best Model Selection

The **XGBoost model** was chosen as the final model based on its high AUC-ROC (80.14%) and accuracy (80.14%). It demonstrated robust performance with fine-tuned parameters:
- **Learning Rate (eta):** 0.1
- **Max Depth:** 5
- **Min Child Weight:** 16
- **Boosting Rounds:** 140

This model balances generalization and accuracy, making it suitable for deployment in production.

## Instructions to Run the Application

### Prerequisites
1. Python 3.9+: Ensure Python is installed on your machine.
2. Pipenv: Install Pipenv for dependency management.
3. Docker (optional): To run the application in a containerized environment.

### Local Setup
1. Clone the repository:
   ```bash
   git clone `git@github.com:okellodaniel/cardio_vasicular_disease_detector.git`
   cd cardio-vascular-disease-detector
   ```

2. Install dependencies:
   ```bash
   pip install pipenv
   pipenv install --system --develop
   ```

3. Run the application:
   ```bash
   uvicorn app.main:app --port 5050
   ```

4. Test locally using the FastAPI docs:
   - Open a browser and navigate to `http://localhost:5050/` for an interactive scalar api docs interface.

---

### Using Docker
1. Build the Docker container:
   ```bash
   docker build -t cardio-detector .
   ```

2. Run the container:
   ```bash
   docker run -p 5050:5050 cardio-detector
   ```

### Cloud Deployment
The application is hosted on `fly.io` and can be accessed via:
- **URL**: [https://cardio-vasicular-disease-detector.fly.dev](https://cardio-vasicular-disease-detector.fly.dev)

## Testing the Application

### **Using Scalar API Docs**
Scalar API Docs is an extension for testing FastAPI applications.

1. **Using cURL**
Send a POST request to test the API:
```bash
curl -X POST "http://localhost:5050/" \
-H "Content-Type: application/json" \
-d '{
  "age": 18300,
  "gender": "male",
  "height": 168,
  "weight": 62,
  "ap_hi": 120,
  "ap_lo": 80,
  "cholesterol": "normal",
  "gluc": "normal",
  "smoke": "no",
  "alco": "no",
  "active": "yes"
}'
```

2.  **Using Python Requests**

```python
import requests

url = "http://localhost:5050/"
data = {
    "age": 18300,
    "gender": "male",
    "height": 168,
    "weight": 62,
    "ap_hi": 120,
    "ap_lo": 80,
    "cholesterol": "normal",
    "gluc": "normal",
    "smoke": "no",
    "alco": "no",
    "active": "yes"
}
response = requests.post(url, json=data)
print(response.json())
```

**Expected Response**
```json
{
  "prediction": 0.2345,
  "cardio": false,
  "prediction%": "23.45%",
  "threshold": 0.5
}
```

---

## Future Improvements

1. **Integration with Healthcare Systems:**
   - Stream real-time data from IoT devices for continuous monitoring.

2. **Model Enhancements:**
   - Explore deep learning models for complex feature interactions.
   - Investigate ensemble learning techniques to improve prediction reliability.

3. **Explainability:**
   - Implement SHAP (SHapley Additive exPlanations) for better interpretability of predictions.

4. **Scalability:**
   - Deploy on Kubernetes to handle high traffic and ensure availability.

5. **User Interface:**
   - Add a web-based dashboard for healthcare professionals to interact with predictions.

---