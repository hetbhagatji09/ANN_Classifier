import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

model=load_model('model.h5')

with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)
with open('scaler.pkl','rb') as f:
    scaler=pickle.load(f)
# print(onehot_encoder_geo.categories_[0])



geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# print(label_encoder_gender.classes_)
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# input_data['Gender']=label_encoder_gender.transform(input_data['Gender'])
final=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)
#print(final)
scaled_data=scaler.transform(final)


if st.button("Predict"):
    st.write("Here are the output of  churn data:")
    # model.predict(scaled_data)
    output=model.predict(scaled_data)[0][0];
    st.write('Predicted Churn:', output)
    if output>0.5:
        st.write('Customer has been marked as churned in the system.')
    else:
        st.write('Customer has not been marked as churned in the system.')
        