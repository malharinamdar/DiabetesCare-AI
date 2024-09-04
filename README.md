# DiabetesCare AI 
Take Charge of Your Health.

DiabetesCare AI - Gemini Enhanced Diabetes Prediction along with a Q & A Chatbot

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Understanding Diabetes](#Understanding-Diabetes)
- [Model Training](#Model-Training)
- [Web App Components](#Web-App-Components)
- [Deployment](#deployment)

## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/malharinamdar/DiabetesCare-AI
   cd DiabetesCare-AI
2. **Install the recquired libraries**
   ```bash
   pip install -r requirements.txt
   
3. **Download the dataset and pre-trained model**
   Place the dataset (`diabetes_prediction_dataset.csv`) and the model (`wowmodel2.pkl`) in the root directory of the project
   
5. **Set up environment variables**
   Create a `.env` file in the root directory and add your Google API key
   ```bash
   GOOGLE_API_KEY=your_google_api_key
6. **Run the App**
   `streamlit run finalapp.py`
   
## Usage
1. Website has been hosted on the internet, refer to [Deployment](#Deployment) section.
2. Running in your local environment involves steps with necessary packages and libraries installed. Use the `.pkl` file
   required and run the following command: 
   1. For `Windows` users: in `command prompt`, type in:  `streamlit run <path to your stremalit .py file>`
   2. For `Mac` users: in `terminal` , type in: `stremalit run <path to your stremalit .py file>`
      
3. Access the app
   
   Open your web browser and navigate to `http://localhost:8501`.
   
## File Structure

- `finalapp.py`: The main script for running the Streamlit web app.
- `requirements.txt`: Lists all the dependencies required for the project.
- `wowmodel2.pkl`: The pre-trained machine learning model.
- `diabetes_prediction_dataset.csv`: The dataset used for predictions and visualizations.
   
## Understanding Diabetes
General breakdown of food ingested in our body is broken down as glucose by help from insulin secreted by pancreas.

Two things take place in case of diabetes:
- pancreas does not make enough insulin
- body's cells resists insulin's effects

### Type 1 diabetes:
Your body attacks insulin-producing cells, so you need external supply of medication (insulin) to survive.
### Type 2 diabetes: 
Your body becomes resistant to insulin or doesn't make enough. Often managed with lifestyle changes and/or medication, resulting in 
high blood glucose.

## Model Training 
Referred to the mentioned research paper while trying to decide on the best fit for paramters and selecting suitable model.
<a href="https://ieeexplore.ieee.org/document/10128216">research paper</a>

### Why Random Forest?
1. **Handling of Large Data:** Efficient with high-dimensional datasets.
2. **Robustness to Overfitting:** Reduces overfitting by averaging the predictions of multiple decision trees.
3. **Handling Mixed Data Types:** Manages both numerical and categorical features smoothly.
4. **Feature Importance:** Provides estimates of feature importance.
5. **Non-linearity:** Captures complex and non-linear relationships in medical data.
   
### Data Preprocessing and Exploratory Data Analysis (EDA)
Conducted comprehensive steps and functions to address the dataset issues, like containing duplicate rows, 
values and missing parameter values.

### Imbalanced datset 
Implemented `SMOTE` (Synthetic Minority Oversampling Technique) to address the imbalanced dataset

### Preprocessing, Model Building and Hyperparameter Tuning
- Conducted one-hot encoding categorical features
- Implemented `Random Forest` with `GridSearchCV` for Hyperparameter Tuning
  
### Results:
- Selected best parameters after cross-validation.
- Evaluated model performance on the test set.
- Achieved high accuracy `0.94` on the test set.
  
## Web App Components
1. **User Authentication**
   
    Users can log in using their name.
    The session state is used to manage user accounts and their prediction history
2. **Data Input**
   
    The sidebar allows users to input various health indicators such as `gender`, `age`,
    `hypertension`, `heart disease`, `smoking history`, `height`, `weight`, `HbA1c level`, and `blood glucose level`.
   
    BMI is calculated automatically based on the height and weight inputs.
3.  **Prediction**
   
    The `predict_button` triggers the prediction function which uses the pre-trained model to predict the likelihood of diabetes.
    The prediction result is displayed to the user.
    
    The pre-trained achieving an accuracy of `94%` on a `100000` large dataset was loaded as a `.pkl` file
    into the `streamlit` code.
4. **Visualisation**
   
   `Seaborn` and `Matplotlib` enables the users to examine the relationship between the features.
   Multiple graphs integrated to provide a comprehensive overview of the `input data` and the `diabetes_prediction_dataset.csv`.

   If diabetic the plot on the graph is a unique circle with a shade of `red`, else the circle plotted is dark shade of `blue`.
5. **Suggestions**
    
   Personalised lifestyle and dietary suggestions, including helpful resources of hospitals in India are provided
   by the integrtaion of `gemini-1.5-flash` LLM model.

   Sutable `safety_settings` and `temperature` was configured along with the **nucleus sampling** of the
   `top_k` and `top_p` temperature

   Kept maximum output of `4096` tokens at a time for the assistance.
6. **Q & A Chatbot**
    
   The website hosts a Q & A Chatbot to answer queries arising by patients. The history of queries entered by the user are saved and
   displayed in the end.
   The chatbot leverages the use of `gemini-1.5-flash` LLM Model.
     
## Deployment
- Deploying on Streamlit Community Cloud
- Website now live at <a href="https://diabetescare-ai-tech.streamlit.app/">DiabetesCare-AI</a>
  
