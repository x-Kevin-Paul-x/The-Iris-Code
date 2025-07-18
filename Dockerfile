# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Using --no-cache-dir to reduce layer size
# Using --trusted-host to avoid SSL issues with PyPI if behind a proxy (remove if not needed)
RUN pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt

# Copy the rest of the application code into the container at /app
# This includes app.py, the templates folder, the models folder, and the data folder
COPY . .

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable for Flask app (optional, can be set in CMD too)
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Run app.py when the container launches
# Use gunicorn for a production-ready WSGI server if desired,
# For this example, we'll use Flask's built-in server, suitable for development/testing.
# For production, consider: CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
CMD ["flask", "run"]

# Ensure the data and models directories are copied.
# If they are generated by the notebook, they need to exist before building the Docker image.
# The notebook iris_eda_and_model_training.ipynb should be run to generate:
# - data/iris.csv (if download_data.py is not run separately)
# - models/iris_classifier_model.pkl
# - models/iris_scaler.pkl
# - models/iris_label_encoder.pkl
