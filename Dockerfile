# Use an official Python runtime as a parent image
FROM python:3.8-slim

WORKDIR /app

# Set the working directory to /app
COPY app.py /app/app.py
COPY models/ /app/models/
COPY templates/ /app/templates/
COPY docker_requirements.txt /app/docker_requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install -r docker_requirements.txt

# Copy files from S3 inside docker
# RUN mkdir /app/models
# RUN aws s3 cp s3://creditcard-project/models/model.joblib /app/models/model.joblib


# Run app.py when the container launches
CMD ["python", "app.py"]