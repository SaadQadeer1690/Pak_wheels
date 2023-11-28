# Pak_wheels
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("cleaned_data_new.csv")
import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# Assuming 'Price' is your target variable and the rest are your features
X = df.drop('Price', axis=1)
y = df['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define numerical and categorical features
numerical_features = ['Year', 'mileage_km', 'engine_capacity']
categorical_features = ['fuel_type', 'gear_type']

# Create transformers for numerical and categorical features
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a column transformer to apply transformers to different feature sets
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the regression model
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())])

# Train the model
model.fit(X_train, y_train)

# Streamlit web application
st.title('Car Price Prediction App')

# Sidebar for user input
st.sidebar.header('User Input')

# User input for numerical features
user_input = {}
for feature in numerical_features:
    user_input[feature] = st.sidebar.number_input(f'Enter {feature}', value=X[feature].median())

# User input for categorical features
for feature in categorical_features:
    user_input[feature] = st.sidebar.selectbox(f'Select {feature}', X[feature].unique())

# Make predictions
input_df = pd.DataFrame([user_input])
predicted_price = model.predict(input_df)

# Display the predicted price
st.subheader('Predicted Car Price:')
st.write(predicted_price[0])

