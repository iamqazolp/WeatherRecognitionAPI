# Weather Classification API

A PyTorch-based convolutional neural network (CNN) deployed as a REST API using FastAPI. The model classifies images into one of four weather conditions: Cloudy, Rain, Shine, or Sunrise.

## Overview

This project demonstrates fine-tuning a pre-trained ResNet18 model on a custom image dataset and serving the inference logic through a web API. It separates the model training, inference, and routing logic, and is containerized for consistent deployment.

## Project Structure

```text
.
├── data_preprocessing.py   # Data loading and reorganizing
├── finetuning.ipynb        # Training loop and validation
├── inference.py            # Inference helper functions
├── app.py                  # FastAPI server and endpoint routing
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container configuration
└── training.png            # Plot of training/validation loss and accuracy
```

## Model Details
Architecture: ResNet18 (Transfer Learning)

Classes: Cloudy, Rain, Shine, Sunrise

Framework: PyTorch, Torchvision

Techniques: Applied random resized cropping and horizontal flipping during training to mitigate overfitting on a limited dataset.

## Installation and Setup
### 1.Data Acquisition
#### 1.Download the dataset from [Kaggle](https://www.kaggle.com/datasets/pratik2901/multiclass-weather-dataset)
#### 2.Unzip the data into the project root directory.
#### 3.Run the preprocessing script to prepare the directory structure
```
python data_preprocessing.py
```
### API Setup
#### Option 1: Using Docker )
Make sure Docker is installed and running on your system.

Build the Docker image:

```bash
docker build -t weather-api .
```
Run the container:
```bash
docker run -p 8000:8000 weather-api
```
#### Option 2: Local Python Environment
Install the required dependencies:

```bash
pip install -r requirements.txt
```
Start the API server:

```bash
uvicorn app:app --reload
```
### Usage
Once the server is running, the API will be available at http://127.0.0.1:8000.

Interactive UI:
Navigate to http://127.0.0.1:8000/docs to use the built-in Swagger interface. You can upload an image and test the endpoint directly from your browser.

Testing via cURL:

```bash
curl -X 'POST' \
  '[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@path_to_your_test_image.jpg'
```
Example Response:

```json
{
  "filename": "test_image.jpg",
  "prediction": "Shine",
  "probability": "98.50"
}
```
