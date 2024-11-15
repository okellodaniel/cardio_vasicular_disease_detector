# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

ENV DATASET_PATH=../dataset/cardio_vascular_disease_dataset.csv
ENV OUTPUT_FILE=model.bin

# Set the working directory
WORKDIR /app

# Install pipenv and any other dependencies
RUN pip install --upgrade pip && pip install pipenv

# Copy only the Pipfile and Pipfile.lock first to leverage Docker cache
COPY Pipfile Pipfile.lock ./

# Install Python dependencies in system-wide packages
RUN pipenv install --system --deploy

# Copy the rest of the application code

COPY . .

# Expose the application port
EXPOSE 5050

# Run the application using Gunicorn
CMD ["gunicorn","-c","./config/gunicorn_config.py", "predict:app"]