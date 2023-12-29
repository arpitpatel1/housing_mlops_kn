# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory to /app
COPY . /app
WORKDIR /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Copy files from S3 inside docker
# RUN mkdir /app/models
# RUN aws s3 cp s3://creditcard-project/models/model.joblib /app/models/model.joblib


# Run app.py when the container launches
CMD ["python", "app.py"]