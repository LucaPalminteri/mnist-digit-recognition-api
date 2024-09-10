# MNIST Model Training and Serving

This project demonstrates how to train a neural network model on the MNIST dataset and serve it using an API. The code is written in Python and utilizes packages like `torch`, `torchvision`, and `FastAPI`.

## Features

- **Training a Neural Network**: A Convolutional Neural Network (CNN) is used to classify MNIST digits.
- **Model Persistence**: Save and load the trained model.
- **API for Prediction**: A REST API built with FastAPI is used to serve the model for predictions on new data.
- **Multiprocessing Support**: The Uvicorn server runs asynchronously to efficiently handle API requests.

## Prerequisites

- Python 3.11 or higher
- pip (Python package manager)

## Setup

1. **Clone the repository**:

    ```bash
    git clone https://github.com/LucaPalminteri/mnist-digit-recognition-api
    cd mnist-digit-recognition-api
    ```

2. **Install dependencies**:
    Create and activate a virtual environment, and install the required packages:

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. **Training the model**:
    If you want to retrain the MNIST model, you can run the training script.

    ```bash
    python train.py
    ```

4. **Running the API**:
    You can start the FastAPI server using Uvicorn:

    ```bash
    uvicorn main:app --reload
    ```

    The API will be available at `http://localhost:8000`.

## Project Structure

- **main.py**: Contains the FastAPI application and the logic for loading the model and making predictions.
- **train.py**: Script for training the MNIST CNN model and saving the trained model.
- **model.py**: Defines the MNIST model architecture.
- **requirements.txt**: Lists all the Python dependencies needed to run the project.
- **mnist.pth**: The trained model weights file.

## API Endpoints

- **POST /predict**: Takes an image (28x28 pixel array) and returns the predicted digit.

Example request body:

```json
{
  "image": [0, 1, 2, ..., 783]
}
```

Example response:

```json

{   "predicted_digit": 7 }
```

## Dependencies

The following Python packages are required for this project:

- **torch**: For building and training the neural network.
- **torchvision**: For datasets and model utilities.
- **FastAPI**: For serving the trained model through an API.
- **Uvicorn**: For running the FastAPI application.

## Model Loading Warning

When loading models with `torch.load()`, make sure to handle the security risks associated with untrusted sources. You should set `weights_only=True` to avoid loading unwanted code from pickled files.

## Troubleshooting

- **Deprecation warnings**: You may encounter warnings related to the use of deprecated arguments. Specifically:
  - The `pretrained` argument is deprecated. Use `weights` instead.
  - Use `torch.load(..., weights_only=True)` to avoid future security risks.
- **Model loading errors**: If the model fails to load, ensure that the model architecture matches the saved state dictionary.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

### `requirements.txt`

torch>=1.13.0 torchvision>=0.14.0 fastapi>=0.78.0 uvicorn>=0.18.0
