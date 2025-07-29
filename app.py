# app.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and encoder
model = joblib.load("models/model.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")


st.title("🧠 Predict Personality Type")
st.write("Fill out the form below to predict if the person is an Introvert or Extrovert.")

# Input fields
time_spent_alone = st.number_input("Time spent alone (hours)", min_value=0.0, max_value=24.0, value=3.0)
stage_fear = st.selectbox("Stage fear", options=["Yes", "No"])
social_event_attendance = st.number_input("Social event attendance", min_value=0.0, value=5.0)
going_outside = st.number_input("Going outside (days/week)", min_value=0.0, value=4.0)
drained_after_socializing = st.selectbox("Drained after socializing", options=["Yes", "No"])
friends_circle_size = st.number_input("Friends circle size", min_value=0.0, value=5.0)
post_frequency = st.number_input("Post frequency", min_value=0.0, value=3.0)

# Encode binary categorical fields
stage_fear_encoded = 1 if stage_fear == "Yes" else 0
drained_encoded = 1 if drained_after_socializing == "Yes" else 0

# Prepare input DataFrame
input_data = pd.DataFrame([[
    time_spent_alone,
    stage_fear_encoded,
    social_event_attendance,
    going_outside,
    drained_encoded,
    friends_circle_size,
    post_frequency
]], columns=[
    "Time_spent_Alone",
    "Stage_fear",
    "Social_event_attendance",
    "Going_outside",
    "Drained_after_socializing",
    "Friends_circle_size",
    "Post_frequency"
])

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    # Decode manually since 'Personality' encoder might not exist in label_encoders.pkl
    if prediction == 0:
        personality = "Extrovert"
    elif prediction == 1:
        personality = "Introvert"
    else:
        personality = f"Unknown ({prediction})"
    st.success(f"🧑 This person is likely an **{personality}**.")
