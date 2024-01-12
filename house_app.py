import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px

with open('preprocessed.pkl', 'rb') as file:
    pp = pickle.load(file)

with open('xg.pkl', 'rb') as f:
    xgb = pickle.load(f)

# Function to get user input features
def features():
    st.sidebar.header('User Input Features')

    area_type = st.sidebar.selectbox('Type of Area', ['Super Built-up Area', 'Built-up Area', 'Plot Area', 'Carpet Area'])
    location = st.sidebar.selectbox('Location', ['Whitefield', 'Sarjapur Road', 'Electronic City', 'Kanakpura Road'])
    size = st.sidebar.slider('BHK', 1, 10, 1)
    total_sqft = st.sidebar.slider('Area of the house (in sqft)', 500, 50000, 100)
    bath = st.sidebar.slider('Number of Bathrooms', 1, 5, 1)
    balcony = st.sidebar.slider('Number of Balcony', 1, 5, 1)
    availability_mapped = st.sidebar.slider('Ready to move', 0, 1)

    data = {'area_type': area_type, 'location': location, 'size': size, 'total_sqft': total_sqft,
            'bath': bath, 'balcony': balcony, 'availability_mapped': availability_mapped}

    input_vars = pd.DataFrame(data, index=[0])
    return input_vars

# Function to transform user input and get predicted price
def predict_price(input_data):
    df_tf = pp.transform(input_data)
    predicted_price = xgb.predict(df_tf)[0]
    return predicted_price

# Function to simulate data
def simulate_data():
    return pd.DataFrame({
        'location': np.random.choice(['Whitefield', 'Sarjapur Road', 'Electronic City', 'Kanakpura Road'], 1000),
        'price': np.random.normal(1000000, 50000, 1000)
    })

# Streamlit app code
st.title('Bengaluru House Price Predictor App')

# Get user input features
df = features()

# Predict house price
predicted_price = predict_price(df)

# Display predicted price
st.subheader('Predicted House Price:')
st.write(f'₹ {predicted_price:,.2f}')

# Show data only if the user wants to
if st.checkbox('Show Input Data'):
    st.subheader('User Input Data:')
    st.write(df)

# Display distribution of predicted prices using a violin plot from Plotly Express
st.subheader('Distribution of Predicted Prices')
st.warning('This is a simulated distribution and not based on actual data.')

# Simulate a distribution of predicted prices (replace with actual data if available)
simulated_prices = np.random.normal(predicted_price, 50000, 1000)

# Create a violin plot using Plotly Express
fig = px.violin(y=simulated_prices, box=True, points="all", title="Distribution of Predicted Prices")
fig.update_layout(yaxis_title="Predicted Prices")
st.plotly_chart(fig)

# Show average prices for different locations using an interactive bar chart
st.subheader('Average Prices for Different Locations')
st.warning('This is a simulated dataset.')

# Simulate a dataset with average prices for different locations
data = simulate_data()
average_prices = data.groupby('location')['price'].mean().reset_index()

# Create an interactive bar chart using Plotly Express
bar_chart = px.bar(
    average_prices,
    x='location',
    y='price',
    color='location',  # Different colors for each location
    title='Average Prices for Different Locations',
    labels={'price': 'Average Price'},
    height=500
)

bar_chart.update_layout(xaxis_title='Location', yaxis_title='Average Price')
st.plotly_chart(bar_chart)

# Provide additional information or instructions
st.info('Explore different input values using the sidebar to see how they affect the predicted house price.')
