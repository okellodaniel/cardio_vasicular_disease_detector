import logging
import io
import re
import contextlib
import requests
import pandas as pd
import numpy as np
import seaborn as sns
import xgboost as xgb
import random
import pickle

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, classification_report, confusion_matrix

# Initialize logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# File and Parameter Settings
OUTPUT_FILE = 'model.bin'

# Mappings for categorical values
CHOL_LEVELS = {1: 'normal', 2: 'above_normal', 3: 'well_above_normal'}
GLUC_LEVELS = {1: 'normal', 2: 'above_normal', 3: 'well_above_normal'}
GENDER_VALUES = {1: 'female', 2: 'male'}
YES_NO_VALUES = {0: 'no', 1: 'yes'}

# Data Preparation


def load_and_prepare_data(filepath):
    """
    Load data, map categorical values, and drop irrelevant columns.
    Args:
        filepath (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    logging.info("Loading and preparing data...")
    df = pd.read_csv(filepath, sep=';')
    df['cholesterol'] = df['cholesterol'].map(CHOL_LEVELS)
    df['gluc'] = df['gluc'].map(GLUC_LEVELS)
    df['gender'] = df['gender'].map(GENDER_VALUES)
    df['smoke'] = df['smoke'].map(YES_NO_VALUES)
    df['active'] = df['active'].map(YES_NO_VALUES)
    df['alco'] = df['alco'].map(YES_NO_VALUES)
    df = df.drop('id', axis=1)
    return df


def split_data(df):
    """
    Split the data into train, validation, and test sets.
    Args:
        df (pd.DataFrame): The full DataFrame.
    Returns:
        dict: Dictionary containing splits of the data and target variables.
    """
    logging.info("Splitting data into train, validation, and test sets...")
    df_full_train, df_test = train_test_split(
        df, test_size=0.2, random_state=1, shuffle=True)
    df_train, df_val = train_test_split(
        df_full_train, test_size=0.25, random_state=1, shuffle=True)

    y_train = df_train['cardio'].values
    y_val = df_val['cardio'].values
    y_test = df_test['cardio'].values
    y_full_train = df_full_train['cardio'].values

    df_full_train = df_full_train.drop('cardio', axis=1)
    df_train = df_train.drop('cardio', axis=1)
    df_val = df_val.drop('cardio', axis=1)
    df_test = df_test.drop('cardio', axis=1)

    return {
        "df_full_train": df_full_train, "df_train": df_train, "df_val": df_val, "df_test": df_test,
        "y_full_train": y_full_train, "y_train": y_train, "y_val": y_val, "y_test": y_test
    }


def one_hot_encode(data_splits):
    """
    Perform one-hot encoding on the categorical features.
    Args:
        data_splits (dict): Dictionary containing train, validation, and test sets.
    Returns:
        dict: Dictionary with transformed feature matrices and target variables.
    """
    logging.info("One-hot encoding categorical variables...")
    dv = DictVectorizer(sparse=False)

    X_full_train = dv.fit_transform(
        data_splits["df_full_train"].to_dict(orient='records'))
    X_train = dv.transform(data_splits["df_train"].to_dict(orient='records'))
    X_val = dv.transform(data_splits["df_val"].to_dict(orient='records'))
    X_test = dv.transform(data_splits["df_test"].to_dict(orient='records'))

    return {
        "X_full_train": X_full_train, "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_full_train": data_splits["y_full_train"], "y_train": data_splits["y_train"],
        "y_val": data_splits["y_val"], "y_test": data_splits["y_test"]
    }


def train_xgboost_model(X, y):
    """
    Train the final XGBoost model with specified parameters.
    Args:
        X (array-like): Training features.
        y (array-like): Training target variable.
    Returns:
        xgb.Booster: Trained XGBoost model.
    """
    logging.info("Training XGBoost model...")
    dtrain = xgb.DMatrix(X, label=y)

    xgb_params = {
        'eta': 0.1,
        'max_depth': 5,
        'min_child_weight': 16,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'seed': 1,
    }
    model = xgb.train(xgb_params, dtrain, num_boost_round=500, verbose_eval=10)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using ROC-AUC, accuracy, precision, recall, and confusion matrix.
    Args:
        model (xgb.Booster): Trained model.
        X_test (array-like): Test features.
        y_test (array-like): Test target variable.
    """
    logging.info("Evaluating model...")
    dtest = xgb.DMatrix(X_test)
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)

    # AUC-ROC
    auc = roc_auc_score(y_test, y_pred_proba)
    logging.info(f"AUC-ROC Score: {auc:.4f}")

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Accuracy: {accuracy:.4f}")

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()


def export_model(model):
    with open(OUTPUT_FILE, 'wb') as f_out:
        pickle.dump(model)


def main():
    # Load and prepare the data
    df = load_and_prepare_data(
        '../dataset/cardio_vascular_disease_dataset.csv')

    # Split the data
    data_splits = split_data(df)

    # One-hot encode features
    encoded_data = one_hot_encode(data_splits)

    # Train model on full training set
    model = train_xgboost_model(
        encoded_data["X_full_train"], encoded_data["y_full_train"])

    # Evaluate the model on the test set
    evaluate_model(model, encoded_data["X_test"], encoded_data["y_test"])

    export_model(model)


if __name__ == "__main__":
    main()
