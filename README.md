# DiabetesCare-AI 
personalised diabetes prediction along with a Q & A Chatbot

   
## File Structure

- `finalapp.py`: The main script for running the Streamlit web app.
- `requirements.txt`: Lists all the dependencies required for the project.
- `wowmodel2.pkl`: The pre-trained machine learning model.
- `diabetes_prediction_dataset.csv`: The dataset used for predictions and visualizations.
   

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
     
