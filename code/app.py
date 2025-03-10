import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the model and scaler from the pickle files
model = pickle.load(open('knn_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

def data_preprocess(features):
    # Create a DataFrame with the input features
    df = pd.DataFrame([features], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    
    # Scale numeric features
    numeric_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    
    # Encode Age feature
    df['Age_Category_Young'] = ((df['Age'] >= 21) & (df['Age'] < 45)).astype(int)
    
    # Encode BMI feature
    df['BMI_Category_Healthy'] = ((df['BMI'] >= 18.5) & (df['BMI'] < 25)).astype(int)
    df['BMI_Category_Overweight'] = ((df['BMI'] >= 25) & (df['BMI'] < 30)).astype(int)
    df['BMI_Category_Obese'] = (df['BMI'] >= 30).astype(int)
    
    # Ensure all required columns are present
    feature_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'DiabetesPedigreeFunction', 
                    'BMI_Category_Healthy', 'BMI_Category_Overweight', 'BMI_Category_Obese', 
                    'Age_Category_Young', 'Pregnancies']
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    
    # Reorder the columns to match the model's expectations
    df = df[feature_cols]
    
    return df

def predict(features):
    prediction = model.predict(features)
    if prediction[0] == 0:
        return "The person is HEALTHY"
    else: 
        return "The person is DIABETIC and needs medical assisstance"

# Streamlit UI
st.title('DIABETES PREDICTION SYSTEM')
st.write('Diabetes is a rising concern in this modern world, It affects people of all ranges from infants to elderly people.')
st.write('It affects millions worldwide and leads to severe health complications')
st.write('The main purpose of this project is to develop a predictive model using machine learning and data science that could predict diabetes with a reasonable accuracy.')
st.write('Please fill in the details of the person under consideration in the left sidebar and click on the button below!')

age = st.sidebar.number_input("Age in Years", 1, 150, 25, 1)
pregnancies = st.sidebar.number_input("Number of Pregnancies", 0, 20, 0, 1)
glucose = st.sidebar.slider("Glucose Level", 0, 200, 25, 1)
skinthickness = st.sidebar.slider("Skin Thickness", 0, 99, 20, 1)
bloodpressure = st.sidebar.slider('Blood Pressure', 0, 150, 80, 1)
insulin = st.sidebar.slider("Insulin", 0, 850, 80, 1)
bmi = st.sidebar.slider("BMI", 0.0, 67.5, 31.4, 0.1)
dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.000, 2.500, 1.471, 0.001)

features = [pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age]

if st.button('Find Diabetes Status'):
    preprocessed_features = data_preprocess(features)
    result = predict(preprocessed_features)
    st.write(result)