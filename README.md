# DiabetesCare AI 
Take Charge of Your Health: 

DiabetesCare AI - an AI Enhanced Diabetes Prediction and Gemini Driven Assistance Companion


### Understanding Diabetes

General breakdown of food ingested in our body is broken down as glucose by help from insulin secreted by pancreas.

Two things take place in case of diabetes:
1. pancreas does not make enough insulin
2. body's cells resists insulin's effects

#### Type 1 diabetes:
Your body attacks insulin-producing cells, so you need external supply of medication (insulin) to survive.
#### Type 2 diabetes: 
Your body becomes resistant to insulin or doesn't make enough. Often managed with lifestyle changes and/or medication, resulting in 
high blood glucose.

#### Deciding Parameters

After analysing a couple of research papers of IEEE, it resulted in deciding the following 8 parameters as our 
inputs upon which the ml model would predict the presence of diabetes in the patients:

<a href="https://ieeexplore.ieee.org/document/10128216">research paper</a>

1. gender (male or female)
2. age
3. hypertension
4. heart disease
5. smoking history
6. haemoglobin level
7. bmi
8. glucose




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
