# Use an official Python runtime as a parent image
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY conda.yaml .

# Create the environment
RUN conda env create -f conda.yaml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Copy the current directory contents into the container at the working directory
COPY . .

# Define environment variable