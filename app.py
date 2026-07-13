import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import requests
import streamlit as st
from src.lstm import LSTMModel
from typing import List

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# print(BASE_DIR)

device = "mps" if torch.mps.is_available() else "cpu"

@st.cache_resource
def load_items():
    scaler = joblib.load('models/scaler.pkl')
    with open(os.path.join(BASE_DIR,  'models', 'lstm_best_params.json')) as f:
        best_parameters = json.load(f)
    model_parameters = torch.load(os.path.join(BASE_DIR, 'models', 'lstm_model.pt'), map_location='cpu', weights_only=True)
    model = LSTMModel(inputSize=scaler.n_features_in_, hiddenSize=best_parameters["hidden_size"], numLayers=best_parameters["num_layers"], dropOut=best_parameters["dropout"]).to(device)
    model.load_state_dict(model_parameters)
    model.eval()
    return model, best_parameters, scaler

model, best_parameters, scaler = load_items()

st.set_page_config(page_title="Aircraft Engine Lifecycle Prediction", layout="centered")

st.title("Aircraft Engine Lifecycle Predictor", text_alignment="center", width="stretch")
st.markdown("<h3 style='text-align: center;'>How many lifecycles do your engines have remaining?</h3>", unsafe_allow_html=True)

test_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'features', 'test_features.csv'))

downloadButton = st.sidebar.download_button(label="Download Test CSV file", data=test_df.to_csv(index=False), file_name="engine_test_data.csv", mime="text/csv")

st.sidebar.header(body="Project Information:")

st.sidebar.subheader(body="Instructions")

st.sidebar.markdown(body="""
                      1. Download the sample CSV files
                      2. Upload it using the file uploader
                      3. View the predictions and risk levels
                      
                      Note:
                        - Must contain engine_id and cycle columns
                        - Must contains the 58 sensor feature columns
                        - Must be in the same format as the training data
                    """, unsafe_allow_html=True)

st.sidebar.subheader(body="Risk Levels")

st.sidebar.markdown(body="""
                      - **High**: - fewer than 30 flights remaining
                      - **Medium**: - 30 to 60 flights remaining
                      - **Low**: - more than 60 flights remaining
                    """, unsafe_allow_html=True)

st.sidebar.info(body=f"Model: LSTM | RMSE: {round(18.108158111572266, 2)} cycles")

uploaded_file = st.file_uploader(label="Upload Engine Sensor CSV file", type=["csv"], help="Please upload a CSV file under 200MB.")

all_columns = test_df.columns.tolist()
excluded_columns = ['engine_id', 'cycle', 'RUL']
selected_columns = [column for column in all_columns if column not in excluded_columns]

if uploaded_file is not None:
    engine_sensor_df = pd.read_csv(uploaded_file)
    st.subheader(body="Engine Sensor CSV File", text_alignment="center")
    sequence_length = best_parameters["sequence_length"]
    
    _, col2, col3, _ = st.columns([3, 2, 2, 3])
    
    with col2:
        st.metric(label="Engines Detected", value=engine_sensor_df['engine_id'].unique().shape[0], label_visibility="visible")
        
    with col3:
        st.metric(label="Flights Recorded", value=engine_sensor_df.shape[0], label_visibility="visible")
      
    input_data = np.array(engine_sensor_df.values)
    num_features = scaler.n_features_in_
    
    prediction_responses = []
    
    for i in range(engine_sensor_df['engine_id'].unique().min(), engine_sensor_df['engine_id'].unique().max() + 1):
      engine_test = engine_sensor_df[engine_sensor_df['engine_id'] == i].tail(sequence_length)[selected_columns]
      input_data = np.array(engine_test.values)
      num_features = scaler.n_features_in_

      if len(input_data) < sequence_length:
        missing = sequence_length - len(input_data)
        padding = np.zeros((missing, num_features))
        input_flight_sequence_data = np.vstack([padding, input_data])
        real_input = {"flight_sequence": input_flight_sequence_data.tolist() }
                
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=real_input
        )
        json_response = response.json()
        prediction_responses.append({
            "engine_id": i,
            "predicted_rul": json_response["predicted_rul"],
            "risk_level": json_response["risk_level"]
        })
      else:
        real_input = {"flight_sequence": engine_test.values.tolist()}
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=real_input
        )
        json_response = response.json()
        prediction_responses.append({
            "engine_id": i,
            "predicted_rul": json_response["predicted_rul"],
            "risk_level": json_response["risk_level"]
        })
    
    results_df = pd.DataFrame(prediction_responses)
    
    results_df.loc[results_df['predicted_rul'] < 0, 'predicted_rul'] = 0
    
    left_col, mid_col, right_col = st.columns([1, 1, 1])
        
    high_risk_engines = results_df[results_df['risk_level'] == "high"]
    medium_risk_engines = results_df[results_df['risk_level'] == "medium"]
    low_risk_engines = results_df[results_df['risk_level'] == "low"]
    average_predicted_rul = results_df['predicted_rul'].mean()
    
    index = high_risk_engines.values[:, 1].argmin()
    
    with left_col:
        st.error(f"High Risk Engines: {high_risk_engines.shape[0]}")
    
    with mid_col:
        st.warning(f"Medium Risk Engines: {medium_risk_engines.shape[0]}")
    
    with right_col:
        st.success(f"Low Risk Engines: {low_risk_engines.shape[0]}")
    
    _, b, c, _ = st.columns([1, 2, 2, 1])
    
    with b:
        st.metric(label="Average predicted RUL: ", value=f"{round(average_predicted_rul, 1)} cycles", label_visibility="visible")
    
    with c:
        st.metric(label="Lowest Engine RUL: ", value=round(high_risk_engines.values[index, :][1], 2), label_visibility="visible")
    
    high_risk_col = st.columns([1])[0]
    
    with high_risk_col:
        if high_risk_engines.shape[0] > 0:          
          if high_risk_engines.shape[0] == 1:
              st.error(f"1 engine requires immediate attention: Engine {high_risk_engines.values[0, 0]}")
          else:
              engines = [str(engine) for engine in list(high_risk_engines.values[:, 0])]
              st.error(f"{high_risk_engines.shape[0]} engines require immediate attention: Engines {', '.join(engines)}")
    
    st.table(results_df.sort_values(by='predicted_rul'))
    
    average_cycles_per_engine = engine_sensor_df.groupby('engine_id')['cycle'].count().mean()

    st.write(f"Average cycles per engine: {int(round(average_cycles_per_engine, 1))}")
  
    
    # Statistics to show
        
        # Fleet level
          # Average predicted RUL across all engines - after predictions run
          # Minimum predicted RUL - which engine is closest to failure
        
        # For the results section (after predictions) show:
          # Average predicted RUL
          # Number of high risk engines
          # Number of medium risk engines
          # Engine with lowest RUl - most urgent maintenance needed
      
      

# 2. Build the file upload section (Let users upload sensor CSV data)
    # Add a file uploader using st.file_uploader() that accepts CSV files
    # When a file is uploaded, read it into a pandas DataFrame
    # Show a preview of the uploaded data using st.dataframe()
    # Show the number of engines and cycles detected in the uploaded file
    # Add a sidebar with instructions explaining what CSV format to upload
  # Use your test_features.csv as the sample file users can download and upload -- it's already in the right format with the right columns.
  
# 4. Displays results table (Show predictions for all engines)
    # Show a summary table with engine_id, predicted RUl and risk level for all engines
    # Color code the risk level - red for high, orange for medium, green for low
    # Show summary metrics at the top - number of high risk engines, average RUL
    # Use st.metric() to display key numbers prominently
  # st.metric() displays a large number with a label - perfect for showing "3 engines at high risk" or "Average RUL: 67 cycles" 
  # prominently at the top of the page.

# Add RUL visualization (Plot predicted RUL for a selected engine)
    # Add a dropdown using st.selectbox() to let users pick an engine
    # Plot that engine's sensor readings over time using st.line_chart()
    # Show the predicted RUL prominently for the selected engine
    # Add a horizontal line at RUL=30 showing the danger threshold
  # This interactive chart is what makes the demo compelling - users can explore individual engines rather than just seeing a static table.
  
# Test locally (Run and verify everything works)
    # Run with: streamlit run app.py from your project root
    # Upload your test_features.csv as a test file
    # Confirm predictions look reasonable for all engines
    # Check the risk level color display correctly
    # Record a short screen GIF of the demo for your README
    # Screenshot the results table and chart
  # Use a screen recording tool like Quicktime on Mac to record a short demo GIT. 
  # This goes in your README and is the single most impactful portfolio addition you can make

# Deploy to Streamlit Cloud (Get a public URL for your resume)
    # Push your latest code to GitHub
    # Go to share.streamlit.io and sign in with GitHub
    # Click "New app" and select your repository
    # Set the main file path to app.py
    # Add a requirements.txt that Streamlit Cloud can install from
    # Deploy and wait 2-3 minutes for it to build
    # Copy the public URL and add it your README and resume
  # Streamlit Cloud runs on Linux - make sure you don't use MPS device in app.py since MPS is Mac only. Use CPU for the deployed version.