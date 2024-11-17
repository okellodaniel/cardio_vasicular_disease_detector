# # Use an official Python runtime as a parent image
# FROM python:3.10-slim

# # Set environment variables
# ENV PYTHONDONTWRITEBYTECODE 1
# ENV PYTHONUNBUFFERED 1

# ENV DATASET_PATH=../dataset/cardio_vascular_disease_dataset.csv
# ENV OUTPUT_FILE=model.bin

# # Set the working directory
# WORKDIR /app

# # Install pipenv and any other dependencies
# RUN pip install --upgrade pip && pip install pipenv

# # Copy only the Pipfile and Pipfile.lock first to leverage Docker cache
# COPY Pipfile Pipfile.lock ./

# # Install Python dependencies in system-wide packages
# RUN pipenv install --system --deploy

# # Copy the rest of the application code

# COPY . .

# # Expose the application port
# EXPOSE 5050

# # Run the application using Gunicorn
# CMD ["gunicorn","-c","./config/gunicorn_config.py", "predict:app"]

# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Environment variables for the model and dataset
ENV DATASET_PATH=../dataset/cardio_vascular_disease_dataset.csv
ENV OUTPUT_FILE=model.bin

# Set the working directory
WORKDIR /app

# Install pip, pipenv and required dependencies
RUN pip install --upgrade pip && pip install pipenv

COPY Pipfile Pipfile.lock ./

# Install Python dependencies in system-wide packages
RUN pipenv install --system --deploy

# Copy the rest of the application code
COPY . .

# Expose the application port
EXPOSE 5050

# Command to run the application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5050"]
