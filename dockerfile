# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /app

# Install system dependencies for build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    libssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pipenv and any other dependencies
RUN pip install --upgrade pip && pip install pipenv

# Copy only the Pipfile and Pipfile.lock first to leverage Docker cache
COPY Pipfile Pipfile.lock ./

# Install Python dependencies in system-wide packages
RUN pipenv install --system --deploy

# Copy the rest of the application code
COPY . .

# Expose the application port
EXPOSE 5000

# Run the application using Gunicorn
CMD ["gunicorn", "-c", "config/gunicorn_config.py", "app/predict:predict"]