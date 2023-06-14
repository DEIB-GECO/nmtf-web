# Use the official Python 3.7 image as the base
FROM python:3.7

# Set the working directory inside the Docker image
WORKDIR /app

# Copy the program files to the Docker image
COPY . /app

# Install dependencies
RUN pip install -r requirements.txt

# Expose ports
EXPOSE 5000

# Define the startup command
CMD ["python", "app.py"]

