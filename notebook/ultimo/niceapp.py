import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
import google.generativeai as genai
import pickle

# Configure Gemini API
GOOGLE_API_KEY = "AIzaSyCr109nLhfwS7ozcKEsO20PldcmWHoxgYA"
genai.configure(api_key=GOOGLE_API_KEY)

MODEL_CONFIG = {
  "temperature": 0.2,
  "top_p": 1,
  "top_k": 32,
  "max_output_tokens": 4096,
}

# Safety Settings of Model
safety_settings = [
  {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
  {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
  {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
  {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
]

model = genai.GenerativeModel('gemini-1.5-flash', safety_settings=safety_settings)
columns = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']

# Load the model
with open('/Users/malhar.inamdar/Desktop/streamlitapp/wowmodel.pkl', 'rb') as file:
    saved_model = pickle.load(file)

# Load the dataset
df = pd.read_csv('/Users/malhar.inamdar/Desktop/streamlitapp/diabetes_prediction_dataset.csv')

# Custom CSS for the sidebar
st.markdown(
    """
    <style>
    .css-1d391kg {
        background-color: #d9edf7 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title
st.title('DiabetesCare AI', anchor='title')

# Sidebar for patient data input
st.sidebar.header('Patient Data')
st.subheader('Training Data Stats')
st.write(df.describe())

# Function for user input
def user_report():
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    age = st.sidebar.slider('Age', 0, 110, 30)
    hypertension = st.sidebar.selectbox('Hypertension', [0, 1])
    heart_disease = st.sidebar.selectbox('Heart Disease', [0, 1])
    smoking_history = st.sidebar.selectbox('Smoking History', ['current', 'non-smoker', 'past_smoker'])
    bmi = st.sidebar.slider('BMI', 10, 97, 20)
    HbA1c_level = st.sidebar.slider('HbA1c Level', 3.0, 11.0, 5.0)
    blood_glucose_level = st.sidebar.slider('Blood Glucose Level', 75, 310, 88)

    user_report_data = {
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'smoking_history': smoking_history,
        'bmi': bmi,
        'HbA1c_level': HbA1c_level,
        'blood_glucose_level': blood_glucose_level
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

# Patient data input
user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)

# Prediction function
def dia_predict(adata):
    return saved_model.predict(adata)

user_result = dia_predict(user_data)

# Visualization section
st.title('Visualized Patient Report')

# Dataset split for graphs
m = df.drop(['diabetes'], axis=1)
n = df.iloc[:, -1]

# Color function for results
color = 'blue' if user_result[0] == 0 else '#8B0000'  # Dark red

# Age vs Glucose
st.header('Age vs Glucose')
fig_glucose = px.scatter(df, x='age', y='blood_glucose_level', color='diabetes',
                         title='Age vs Blood Glucose Level',
                         labels={'age': 'Age', 'blood_glucose_level': 'Blood Glucose Level'},
                         color_continuous_scale='Bluered')
st.plotly_chart(fig_glucose)

# Age vs Heart Disease
st.header('Age vs Heart Disease')
fig_bp = px.scatter(df, x='heart_disease', y='age', color='diabetes',
                    title='Age vs Heart Disease',
                    labels={'heart_disease': 'Heart Disease', 'age': 'Age'},
                    color_continuous_scale='Bluered')
st.plotly_chart(fig_bp)

# Age vs Smoking History
st.header('Age vs Smoking History')
fig_st = px.scatter(df, x='age', y='smoking_history', color='diabetes',
                    title='Age vs Smoking History',
                    labels={'age': 'Age', 'smoking_history': 'Smoking History'},
                    color_continuous_scale='Bluered')
st.plotly_chart(fig_st)

# Age vs HbA1c Level
st.header('Age vs HbA1c Level')
fig_i = px.scatter(df, x='age', y='HbA1c_level', color='diabetes',
                   title='Age vs HbA1c Level',
                   labels={'age': 'Age', 'HbA1c_level': 'HbA1c Level'},
                   color_continuous_scale='Bluered')
st.plotly_chart(fig_i)

# Age vs BMI
st.header('Age vs BMI')
fig_bmi = px.scatter(df, x='age', y='bmi', color='diabetes',
                     title='Age vs BMI',
                     labels={'age': 'Age', 'bmi': 'BMI'},
                     color_continuous_scale='Bluered')
st.plotly_chart(fig_bmi)

# Age vs Hypertension
st.header('Age vs Hypertension')
fig_dpf = px.scatter(df, x='age', y='hypertension', color='diabetes',
                     title='Age vs Hypertension',
                     labels={'age': 'Age', 'hypertension': 'Hypertension'},
                     color_continuous_scale='Bluered')
st.plotly_chart(fig_dpf)

# Output result
st.subheader('Your Report:')
output = 'You are not Diabetic' if user_result[0] == 0 else 'You are Diabetic'
st.title(output)

# Suggestions from Gemini API
def generate_suggestion(report_data, user_result):
    input_text = ", ".join([f"{col}: {val}" for col, val in zip(columns, report_data.iloc[0])])
    if user_result[0] == 0:
        return "You are not diabetic but still keep a healthy lifestyle to prevent future diagnosis."
    else:
        input_string = f"Give me personalised lifestyle and dietary suggestions for a patient with diabetes as per the data given: {input_text}"
        try:
            response = model.generate_content(input_string)  # Adjust the method name and parameters accordingly
            return f"Patient {response.text}"
        except AttributeError:
            # If generate_content is not the right method, try a general method call
            response = model.generate(input_string)
            return f"Patient {response.text}"

st.subheader('Suggestions:')
suggestions = generate_suggestion(user_data, user_result)
st.write(suggestions)
