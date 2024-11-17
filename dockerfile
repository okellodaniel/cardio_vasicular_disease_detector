
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

COPY . .

# Install Python dependencies in system-wide packages
RUN pipenv install --system --deploy

# Expose the application port
EXPOSE 5050

# Command to run the application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5050"]
