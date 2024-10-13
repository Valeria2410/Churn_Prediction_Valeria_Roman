import pandas as pd
import numpy as np


def data_preprocessing(df_new):

  # STEP 1 -> Delete customer ID because is not important in the analysis 
  df_new = df_new.drop(columns=['customerID'])

  # STEP 2 -> It's important to strip any leading/trailing spaces and convert empty values to NaN
  df_new['TotalCharges'] = df_new['TotalCharges'].replace(" ", np.nan)

  # STEP 3 -> Now we are going to convert with this line of code the 'TotalCharges' column to float
  df_new['TotalCharges'] = df_new['TotalCharges'].astype(float)

  # STEP 4 -> We are going to delete the nulls in this column 
  df_new = df_new.dropna(subset=['TotalCharges'])

  # STEP 5 -> We are going to drop the duplicate rows
  df_new = df_new.drop_duplicates()

  # STEP 6 -> We drop this column from the model 
  df_new = df_new.drop('TotalCharges', axis=1)

  df_new['Churn'] = df_new['Churn'].map({'Yes': 1, 'No': 0})


  return df_new