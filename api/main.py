import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from pydantic import BaseModel
import torch
import pandas as pd
import numpy as np
import joblib
from src.lstm import LSTMModel
import json
from typing import List, Literal

app = FastAPI()

device = "mps" if torch.mps.is_available() else "cpu"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

scaler = joblib.load(os.path.join(BASE_DIR, 'models', 'scaler.pkl'))

with open(os.path.join(BASE_DIR, 'models', 'lstm_best_params.json')) as f:
    best_parameters = json.load(f)

model_parameters = torch.load(os.path.join(BASE_DIR, 'models', 'lstm_model.pt'), weights_only=True)
sequence_length = best_parameters["sequence_length"]
num_features = scaler.n_features_in_

model = LSTMModel(inputSize=num_features, hiddenSize=best_parameters["hidden_size"], numLayers=best_parameters["num_layers"], dropOut=best_parameters["dropout"]).to(device)

model.load_state_dict(model_parameters)

model.eval()

class EngineSequence(BaseModel):
    flight_sequence: List[List[float]]

class PredictionResponse(BaseModel):
    predicted_rul: float # a float representing how many flights the engine has left
    risk_level: str # a string that's either "low", "medium", "high"

@app.get("/health")
def health():
    """This is a GET endpoint that just confirms the API is running. Returns a simple dictionary.""" 
    return { "status": "ok", "model": "LSTM", "sequence_length": sequence_length}
    

@app.post("/predict", response_model=PredictionResponse)
def predict(engine_sequence: EngineSequence):
    """This is a POST endpoint that takes an EngineSequence as input and returns a PredictionResponse""" 
    # 1. Convert the input.flight_sequence to a numpy array of shape (45, 58)
    input_flight_sequence_data = np.array(engine_sequence.flight_sequence).reshape(45, 58)
    print(f"Input shape: {input_flight_sequence_data.shape}")
    
    # 2. Apply scaler.transform() to normalize it
    input_flight_sequence_data_scaled = scaler.transform(input_flight_sequence_data)
    
    # 3. Reshape to (1, 45, 58) - adding the batch dimension
    input_flight_sequence_data_reshaped = input_flight_sequence_data_scaled.reshape(1, 45, 58)
    
    # 4. Convert to a float32 PyTorch tensor and move to device
    input_flight_sequence_tensor = torch.tensor(input_flight_sequence_data_reshaped, dtype=torch.float32).to(device)
    
    # 5. Run through the model inside torch.no_grad()
    with torch.no_grad():
        y_pred = model(input_flight_sequence_tensor)
    
    # 6. Extract the RUL value - call .item() to convert from tensor to Python float, then move to CPU first
    remaining_useful_lifeycles = y_pred.cpu().item()
    # 7. Determine risk level -- high if RUL < 30, medium if RUL < 60, low otherwise
    risk = "high" if remaining_useful_lifeycles < 30 else ("medium" if remaining_useful_lifeycles < 60 else "low")
    # 8. Return PredictionResponse schema
    return PredictionResponse(predicted_rul=remaining_useful_lifeycles, risk_level=risk)