import streamlit as st
import os
import pandas as pd
import joblib
from src.preprocessing.preprocessing import data_preprocessing
import plotly.graph_objects as go


st.set_page_config(page_title="Churn Prediction")


df = pd.read_csv('data/Telco_customer_churn.csv')


catboost_path = 'Catboost'
output_model_filepath = f'models/{catboost_path}_model.pkl'
catboost = joblib.load(output_model_filepath)

def show_page():
    st.image('images/UTS-Logo-Syd.jpg')
    st.title("Client Churn Prediction")
    st.subheader("32513 31005 Advanced Data Analytics Algorithms, Machine Learning")
    st.write(" **Student**: Valeria Roman, **Student Id:** 24896716")
    st.subheader("About the Project")
    st.write("""
    The objective of this app is to predict customer churn for individual telecom subscribers by using a Machine Learning model, the app analyzes customer data to estimate the likelihood of a given customer leaving the company. This prediction helps the company identify at-risk customers and take proactive actions to retain them.
    """)





def churn_pred(df):
    st.subheader("Churn prediction by client")

    list_clients = df['customerID']
    client = st.selectbox('Selec client', list_clients)
    client_df = df[df.customerID == client].reset_index(drop = True)
    client_df = data_preprocessing(client_df)
    st.dataframe(client_df)
    prediction = catboost.predict(client_df)
    churn_probabilities = catboost.predict_proba(client_df)[:, 1] 

    churn_prob = churn_probabilities[0]  
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=churn_prob * 100,
        title={'text': "Churn Probability (%)"},
        gauge={'axis': {'range': [0, 100]},
            'bar': {'color': "darkred" if churn_prob > 0.8 else "orange" if churn_prob > 0.5 else "green"},
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "red"}]
            }
    ))

    st.plotly_chart(fig)
    if churn_probabilities > 0.8:
        st.warning('High Risk: The client is at a high risk of leaving the company. Immediate retention efforts are recommended.')
    elif churn_probabilities > 0.5:
        st.info('Moderate Risk: The client may be at risk of leaving. Consider proactive engagement to prevent churn.')
    else:
        st.success('Low Risk: The client is unlikely to leave the company at this time.')


show_page()
 
churn_pred(df)

 

