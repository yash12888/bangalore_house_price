# App code: https://bangalorehouseprice.streamlit.app/

## Objective: To predict House price in Bangalore given various features

### Description: 
  Conducted Exploratory Data Analysis (EDA) and preprocessed the dataset for better model performance.
  Implemented two pipelines for standard scaling of numerical columns and one-hot encoding of categorical columns.
  Developed a transformer containing both pipelines for seamless data transformation.

### Model Training:
  Trained four regression models (XGBoost, Linear Regression, Random Forest, Gradient Boost) to predict house prices.
  Selected XGBoost as the final model based on superior performance metrics (RMSE and MAPE).

### Deployment
  Deployed the predictive model as a web application using the Streamlit framework.
  Utilized pickle for model and preprocessing pipeline serialization.
  Implemented a user-friendly interface with Streamlit for input feature selection and displaying predicted house prices.
