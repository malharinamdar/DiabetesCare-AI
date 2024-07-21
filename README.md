# DiabetesCare AI 
Take Charge of Your Health: 

DiabetesCare AI - an AI Enhanced Diabetes Prediction and Gemini Driven Assistance Companion

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Understanding Diabetes](#Understanding-Diabetes)
- [Model Training](#Model-Training)
- [Web App Components](#Web-App-Components)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## Installation
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/malharinamdar/diabetes.git](https://github.com/malharinamdar/DiabetesCare-AI)
   cd DiabetesCare-AI
2. **Install the recquired libraries**
   ```bash
   pip install -r requirements.txt
3. **Download the dataset and pre-trained model**
   Place the dataset (`diabetes_prediction_dataset.csv`) and the model (`wowmodel2.pkl`) in the root directory of the project
4. **Set up environment variables**
   Create a `.env` file in the root directory and add your Google API key
   ```bash
   GOOGLE_API_KEY=your_google_api_key
## Usage
1. Run the Streamlit app
   ```bash
   streamlit run app.py
2. Access the app
   Open your web browser and navigate to `http://localhost:8501`.
## File Structure
DiabetesCare-AI/
├── finalapp.py/
├── requirements.txt/
├── wowmodel2.pkl/
└── diabetes_prediction_dataset.csv
## Understanding Diabetes

General breakdown of food ingested in our body is broken down as glucose by help from insulin secreted by pancreas.

Two things take place in case of diabetes:
1. pancreas does not make enough insulin
2. body's cells resists insulin's effects

### Type 1 diabetes:
Your body attacks insulin-producing cells, so you need external supply of medication (insulin) to survive.
### Type 2 diabetes: 
Your body becomes resistant to insulin or doesn't make enough. Often managed with lifestyle changes and/or medication, resulting in 
high blood glucose.
## Model Training 
<a href="https://ieeexplore.ieee.org/document/10128216">research paper</a>



## Web App Components
1. User Authentication
    Users can log in using their name.
    The session state is used to manage user accounts and their prediction history
2. Data Input
    The sidebar allows users to input various health indicators such as gender, age,
    hypertension, heart disease, smoking history, height, weight, HbA1c level, and blood glucose level.
   
    BMI is calculated automatically based on the height and weight inputs.
3.  Prediction
    The `predict_button` triggers the prediction function which uses the pre-trained model to predict the likelihood of diabetes.
    The prediction result is displayed to the user.
    
    The pre-trained achieving an accuracy of `94%` on a `100000` large dataset was loaded as a `.pkl` file
    into the `streamlit` code.
4. Visualisation
   `Seaborn` and `Matplotlib` enables the users to examine the relationship between the features.
   Multiple graphs integrated to provide a comprehensive overview of the `input data` and the `diabetes_prediction_dataset.csv`.

   If diabetic the plot on the graph is a unique circle with a shade of red, else the circle plotted is dark shade of blue.
5. Suggestions
   Personalised lifestyle and dietary suggestions, including helpful resources of hospitals in India are provided
   by the integrtaion of `gemini-1.5-flash` LLM model.

   Sutable `safety_settings` and `temperature` was configured along with the **nucleus sampling** of the
   `top_k` and `top_p` temperature

   Kept maximum output of `4096` tokens at a time for the assistance.
     





### Dataset

Previous idea was to utilise the existing PIMA diabetes dataset available as the 
prominent dataset. Issue pertaining to that included:

1. lack of data. 768 samples only.
2. low accuracy.
3. high variance resulting due to lack of data.

Hence the dataset that had important 8 parameters was chosen to train the model containing 100000 samples
achieving an accuracy of 0.94 (or 94%)

### Model

As a general classification problem, logistic regression seemed like a basic choice.
Overturning this decision was due to the ability of random forest to handle large data and undertanding 
complex relationships between the paramters and the output.

Random forest was used due to its higher accuracy and robustness to overfitting, 
along with SMOTE(synthetic minority oversampling technique) due to imbalanced dataset.

### Gemini
Necessary library imports were done which involved the use of importing google.genrativeai
and the use of GOOGLE_API_KEY.
Used Gemini-1.5-flash outputting maximum of 4096 tokens at one request.

### Deploy
Used Streamlit to deploy the model, due to its quick response time and less complexity involved in the
hosting process.
Run the code in your local environment, with necessary packages and libraries installed. Use the .pkl file
required and run the following command: 
1. For Windows users: in command prompt, type in:  streamlit run <path to your stremalit .py file>
2. For Mac users: in terminal , type in: stremalit run <path to your stremalit .py file>
