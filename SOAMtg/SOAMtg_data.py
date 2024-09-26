import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from random import seed
from random import random

def get_data(file):
    df = pd.read_csv(file)

    # Clean up
    
    df.drop(columns=df.columns[0:1], axis=1, inplace=True)
    
    df.rename({'ID.Codes': 'ID', 
               'Readmission.Status': 'Readmit', 
               'DRG.Class': 'DRG_Class', 
               'DRG.Complication': 'DRG_Comp',
               'HCC.Riskscore': 'HCC'}, axis=1, inplace=True)

    df1 = pd.get_dummies(df, columns = ['Gender','Race','DRG_Class','DRG_Comp'], drop_first=True)

    return df1

def create_risk(df):
    # Ensure 'Readmit' and 'ID' are not part of the model input
    df_X = df.drop(['Readmit', 'ID'], axis=1)
    df_y = df['Readmit']

    # Fit the logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(df_X, df_y)

    # Predict probabilities
    df['preds'] = model.predict_proba(df_X)[:, 1]

    return df

def create_treatment(df, p1_treat, p2_treat=None, percentiles_R=None, impact_factors=None):
    seed(1)
    
    # Assign treatments based on p1_treat and optional p2_treat
    if p2_treat is None:
        df['Tx'] = [1 if np.random.rand() < p1_treat else 0 for _ in df.index]
    else:
        r_vals = np.random.rand(len(df))
        df['Tx'] = np.where(r_vals < p1_treat, 1, 
                            np.where(r_vals < (p1_treat + p2_treat), 2, 0))

    def generate_binary(row):
        if row['Tx'] in [1, 2] and row['Readmit'] == 1:  # Apply treatment effect only if treated and initially readmitted
            p1, p2 = percentiles_R[f'T{row["Tx"]}']
            low, mid, high = impact_factors[f'T{row["Tx"]}']

            # Calculate percentiles for R0 values
            cutoffs = np.percentile(df['preds'], [p1, p2])
            multiplier = low if row['preds'] <= cutoffs[0] else (mid if row['preds'] <= cutoffs[1] else high)

            # Determine the outcome based on the multiplier (adjusting readmission status)
            return 1 - int(np.random.rand() < multiplier)
        else:
            return row['Readmit']  # If no treatment or not initially readmitted, keep the original status

    # Apply the function to generate the new 'Readmit_red' column
    df['Readmit_red'] = df.apply(generate_binary, axis=1)

    return df

def summarize_treatment_allocation(df):
    # Group by the 'Tx' column and calculate the count, percentage, and readmission rates
    summary = df.groupby('Tx').agg(
        count=('Tx', 'count'),
        percentage=('Tx', lambda x: len(x) / len(df) * 100),
        readmit_rate=('Readmit', 'mean'),  # Percentage of original readmits
        readmit_red_rate=('Readmit_red', 'mean')  # Percentage of readmits after treatment
    ).reset_index()

    # Rename the 'Tx' values for clarity
    summary['Treatment'] = summary['Tx'].map({0: 'Control', 1: 'Treatment 1', 2: 'Treatment 2'})

    # Sort by 'Tx' before dropping it
    summary = summary.sort_values('Tx')

    # Reorder columns and drop 'Tx'
    summary = summary[['Treatment', 'count', 'percentage', 'readmit_rate', 'readmit_red_rate']]

    # Format the percentage columns
    summary['percentage'] = summary['percentage'].round(2).astype(str) + '%'
    summary['readmit_rate'] = (summary['readmit_rate'] * 100).round(2).astype(str) + '%'
    summary['readmit_red_rate'] = (summary['readmit_red_rate'] * 100).round(2).astype(str) + '%'

    return summary

