# Use the official lightweight Python image
FROM python:3.12-slim

# Set environment variables to prevent Python from writing .pyc files and to ensure output is immediately flushed
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

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