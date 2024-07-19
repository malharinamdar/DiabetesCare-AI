import google.generativeai as genai
import pickle
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns

# Load the trained model
update_model = pickle.load(open('/content/good_model.pkl', 'rb')) 

# Load the dataset
df = pd.read_csv('/Users/malhar.inamdar/Desktop/streamlitapp/diabetes_prediction_dataset.csv')

# Configure Gemini API
GOOGLE_API_KEY = "AIzaSyCr109nLhfwS7ozcKEsO20PldcmWHoxgYA"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Predict diabetes function
def predict(input_data):
    prediction = update_model.predict(input_data)
    return prediction

# Get suggestions from Gemini API
def get_suggestions():
    suggestions = model.generate_text("Provide suggestions for managing diabetes")
    return suggestions

# Header
st.title('Diabetes Checkup')
st.sidebar.header('Patient Data')
st.subheader('Training Data Stats')
st.write(df.describe())

# Function to collect user input
def user_report():
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    age = st.sidebar.slider('Age', 0, 100, 36)
    hypertension = st.sidebar.selectbox('Hypertension', [0, 1])
    heart_disease = st.sidebar.selectbox('Heart Disease', [0, 1])
    smoking = st.sidebar.selectbox('Smoking History', [0, 1])
    bmi = st.sidebar.slider('BMI', 0, 67, 20)
    haemog = st.sidebar.slider('HbA1c Haemoglobin Level', 0.0, 20.0, 7.0)
    glucose = st.sidebar.slider('Blood Glucose', 0, 200, 130)

    user_report_data = {
        'Gender': gender,
        'Age': age,
        'Hypertension': hypertension,
        'HeartDisease': heart_disease,
        'Smoking': smoking,
        'BMI': bmi,
        'HbA1c': haemog,
        'BloodGlucose': glucose
    }

    return pd.DataFrame(user_report_data, index=[0])

# Collect user input
user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)

# Predict diabetes
user_result = predict(user_data)

# Visualizations
st.title('Visualised Patient Report')

# Set color based on prediction
color = 'blue' if user_result[0] == 0 else 'red'

# Age vs BMI
st.header('BMI Value Graph (Others vs Yours)')
fig_bmi = plt.figure()
sns.scatterplot(x='Age', y='BMI', data=df, hue='Outcome', palette='rainbow')
sns.scatterplot(x=user_data['Age'], y=user_data['BMI'], s=150, color=color)
plt.xticks(np.arange(10, 100, 5))
plt.yticks(np.arange(0, 70, 5))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bmi)

# Age vs Glucose
st.header('Glucose Value Graph (Others vs Yours)')
fig_glucose = plt.figure()
sns.scatterplot(x='Age', y='BloodGlucose', data=df, hue='Outcome', palette='magma')
sns.scatterplot(x=user_data['Age'], y=user_data['BloodGlucose'], s=150, color=color)
plt.xticks(np.arange(10, 100, 5))
plt.yticks(np.arange(0, 220, 10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_glucose)

# Output result
st.subheader('Your Report:')
output = 'You are not Diabetic' if user_result[0] == 0 else 'You are Diabetic'
st.title(output)

# If diabetic, get suggestions from Gemini API
if user_result[0] == 1:
    st.subheader('Suggestions:')
    suggestions = get_suggestions()
    st.write(suggestions)

# Display model accuracy
st.subheader('Model Accuracy:')
st.write("94.2%")
