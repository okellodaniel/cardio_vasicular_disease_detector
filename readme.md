# Cardio Vascular Disease Detector

Cardiovascular disease (CVD) remains the leading cause of death globally, affecting millions annually. Often, CVD progresses undetected until it reaches a critical stage, endangering patients' lives and imposing a significant healthcare burden. Early detection is crucial for enabling timely interventions that improve patient outcomes and reduce healthcare costs.

This project is focused on shifting from reactive treatment to proactive prevention, helping identify cardiovascular risks early and supporting informed decision-making for healthier lives.

---

## Problem Statement

Cardiovascular diseases (CVDs) are the leading cause of mortality globally. Early detection is crucial for effective intervention and management. Utilizing the Cardiovascular Disease dataset from Kaggle, which comprises **70,000 patient records with 13 features**, this project aims to develop a predictive model to assess an individual's risk of developing cardiovascular disease. The model will analyze patient data to provide timely risk assessments, thereby aiding healthcare professionals in making informed decisions and potentially improving patient outcomes.

---
## Project Execution Steps
___
### Exploratory Data Analysis (EDA)

Extensive EDA was conducted to gain insights into the dataset, understand feature distributions, and identify relationships. Key steps included:

1. **Analyzing Missing Values:** Checked for null or missing entries.
2. **Feature Analysis:** Examined the distribution of numerical and categorical features.
3. **Target Variable:** Explored the imbalance in the 'cardio' feature.
4. **Correlation Analysis:** Identified important features like age, cholesterol levels, and blood pressure.
5. **Visualizations:** Used histograms and heatmaps for feature relationships.

---

### Model Training

Several machine learning models were trained and evaluated:

1. **Baseline Models:**
   - Decision Tree Classifier
   - Random Forest Classifier

2. **Advanced Models:**
   - XGBoost

#### Training Process:

1. Split the dataset into training, validation, and test sets.
2. Preprocessed the data using one-hot encoding for categorical variables.
3. Tuned model hyperparameters to improve performance.
4. Selected **XGBoost** as the final model based on AUC-ROC and accuracy scores.

---

## Project Structure

This project is structured as follows:
- A *Model preparation notebook* [`notebook.ipynb`]('./notebooks/notebook.ipynb') to prepare the data, train the various vsrious models and select the final model.
- A *training script* [`model_trainer.py`]('./scripts/model_trainer.py') to prepare the data and train the model. The training logic has been encapsulated in this python script.
- Pipfile - 
- fly.toml - Defines the hosting environment requirements(['fly.io']('fly.io))
- dockerfile - The application is containerized using Docker. The containerization steps are defined in the docker file
- Dataset- the dataset [cardio_vascular_disease_dataset.csv]('./dataset/cardio_vascular_disease_dataset.csv'), and its path is defined in the `.env` `DATASET_PATH`
- GUnicorn configuration - The gunicorn execution requirements are defined in the [`gunicorn_config.py`](./config/gunicorn_config.py) file.

---
### Model Deployment

The trained model has been deployed using Flask. Key features of the deployment:

1. API Endpoint: The Flask app serves predictions via a POST request.
2. Input Format: JSON containing patient data with required health indicators.
3. Output Format: JSON response with:
   - Risk prediction (as a percentage).
   - Binary classification (`True`/`False`).
   - Threshold value (default: `0.5`).

### Deployment Framework:
The app is configured to run using **Gunicorn**, a production-grade WSGI server.
---

## Dependency and Environment Management

All dependencies are managed using a `Pipfile` for reproducibility. To set up the environment:

```bash
pip install pipenv
pipenv install --system --deploy
```

The dependencies include:
- Flask
- Gunicorn
- XGBoost
- Pandas
- NumPy
- Scikit-learn

---
## Run the application
### Local Run
#### Build and Run the Container:
```bash
docker build -t cardio-detector .
docker run -p 5050:5050 cardio-detector
```
### Testing Application
---

## Cloud Deployment

The app has been deployed to the cloud fly.io:
- url - https://cardio-vasicular-disease-detector-hidden-surf-7542.fly.dev
---

## Future Improvements

1. **Model Enhancements:**
   - Explore ensemble methods for improved predictions.
   - Incorporate deep learning models for more complex feature relationships.

2. **Scalability:**
   - Add caching for frequently accessed data.
   - Deploy using Kubernetes for high availability and scalability.

3. **Real-Time Updates:**
   - Integrate the app with real-time patient data streams.

4. **Feature Engineering:**
   - Engineer additional features to improve model performance.

---