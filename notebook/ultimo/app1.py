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

## Safety Settings of Model
safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_NONE"
  }
  
  
]
model = genai.GenerativeModel('gemini-1.5-flash', safety_settings=safety_settings)
columns = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']


# Load the model
with open('/Users/malhar.inamdar/Desktop/streamlitapp/wowmodel.pkl', 'rb') as file:
    saved_model = pickle.load(file)

# Load the dataset
df = pd.read_csv('/Users/malhar.inamdar/Desktop/streamlitapp/diabetes_prediction_dataset.csv')

# HEADINGS
st.title('Diabetes Checkup')
st.sidebar.header('Patient Data')
st.subheader('Training Data Stats')
st.write(df.describe())

# FUNCTION
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

# PATIENT DATA
user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)

# Prediction
def dia_predict(adata):
    return saved_model.predict(adata)

user_result = dia_predict(user_data)

# VISUALISATIONS
st.title('Visualised Patient Report')

#dataset split for graphs
m=df.drop(['diabetes'], axis=1)
n=df.iloc[:, -1]

# COLOR FUNCTION
color = 'blue' if user_result[0] == 0 else 'darkred'


# Age vs Glucose
st.header('Age vs Glucose')
fig_glucose = plt.figure()
sns.scatterplot(x='age', y='blood_glucose_level', data=df, hue='diabetes', color=color)
plt.xticks(np.arange(10, 100, 5))
plt.yticks(np.arange(70, 310, 20))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_glucose)

# Age vs Heart Disease
st.header('Age vs Heart Disease')
fig_bp = plt.figure()
sns.scatterplot(x='heart_disease', y='age', data=df, hue='diabetes', color=color)
plt.xticks(np.arange(0, 2, 1))
plt.yticks(np.arange(0, 110, 10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bp)

# Age vs Smoking History
st.header('Age vs Smoking History')
fig_st = plt.figure()

sns.scatterplot(x='age', y='smoking_history', data=df, hue='diabetes', color=color)
plt.xticks(np.arange(10, 100, 5))
plt.yticks(np.arange(0, 4, 1))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_st)

# Age vs HbA1c Level
st.header('Age vs HbA1c Level')
fig_i = plt.figure()

sns.scatterplot(x='age', y='HbA1c_level', data=df, hue='diabetes', color=color)
plt.xticks(np.arange(10, 100, 5))
plt.yticks(np.arange(0, 12, 1))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_i)

# Age vs BMI
st.header('Age vs BMI')
fig_bmi = plt.figure()

sns.scatterplot(x='age', y='bmi', data=df, hue='diabetes', color=color)
plt.xticks(np.arange(10, 100, 5))
plt.yticks(np.arange(20, 100, 5))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bmi)

# Age vs Hypertension
st.header('Age vs Hypertension')
fig_dpf = plt.figure()

sns.scatterplot(x='age', y='hypertension', data=df, hue='diabetes', color=color)
plt.xticks(np.arange(10, 100, 5))
plt.yticks(np.arange(0, 4, 1))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_dpf)

# OUTPUT
st.subheader('Your Report: ')
output = 'You are not Diabetic' if user_result[0] == 0 else 'You are Diabetic'
st.title(output)

# If diabetic, get suggestions from Gemini API
def generate_suggestion(report_data, user_result):
    input_text = ", ".join([f"{col}: {val}" for col, val in zip(columns, report_data.iloc[0])])
    if user_result[0] == 0:
        return "You are not diabetic but still keep a healthy lifestyle to prevent future diagnosis."
    else:
        input_string = f"Give me lifestyle and dietary suggestions for a patient with diabetes with data as follows: {input_text}"
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
