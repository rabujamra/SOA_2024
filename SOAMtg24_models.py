import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

################################################################
### DATA SETUP 
################################################################

def create_subgroups(df):
    # Ensure 'Readmit' and 'ID' are not part of the model input
    df_X = df.drop(['Readmit', 'ID'], axis=1)
    df_y = df['Readmit']

    # Fit the logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(df_X, df_y)

    # Predict probabilities
    df['preds'] = model.predict_proba(df_X)[:, 1]

    # Define quantiles
    q60 = df['preds'].quantile(0.6)
    q80 = df['preds'].quantile(0.8)

    # Create subgroups
    df['subgroup'] = np.where(
        (df['preds'] >= q60) & (df['preds'] <= q80), 1,
        np.where(df['preds'] > q80, 2, 0)
    )

    return df

def create_treatment(df, p=.4, q=.5, p1=.85, q1=.6):
    # Define a lambda function to apply to each row of the DataFrame
    def generate_binary(row, p, q, p1, q1):
        if row['Tx'] * row['Readmit'] == 1:
            if row['subgroup'] == 1:
                return 1 - int(np.random.rand() < p1 * q1)
            else:
                return 1 - int(np.random.rand() < p * q)
        else:
            return row['Readmit']
    
    # Apply the lambda function to each row of the DataFrame to create Readmit_red
    df['Readmit_red'] = df.apply(lambda row: generate_binary(row, p, q, p1, q1), axis=1)    

    return df

def create_treatment1(df, p=.4, q=.5, p1=.85, q1=.6):
# Define a lambda function to apply to each row of the DataFrame
    def generate_binary_skewed(row, p, q, p1, q1):
        if row['Tx'] * row['Readmit'] == 1:
            if row['subgroup'] == 1:
                # Higher probability of reducing readmission, skewed
                reduction_probability = np.random.uniform(p1*q1, p1*q1 + 0.1)
                return 1 - int(np.random.rand() < reduction_probability)
            else:
                # Higher probability of reducing readmission, skewed
                reduction_probability = np.random.uniform(p*q, p*q + 0.1)
                return 1 - int(np.random.rand() < reduction_probability)
        else:
            return row['Readmit']

    # Apply the lambda function to each row of the DataFrame to create Readmit_red
    df['Readmit_red'] = df.apply(lambda row: generate_binary_skewed(row, p, q, p1, q1), axis=1)

    return df

# TODO: Improve! #
def create_cost(df, Tx, c0=1 , c1=40):
    df['cost'] = df[Tx].apply(lambda x: c1 if x == 1 else c0)

    return df

def calculate_risk_propensity(df, X_columns, Tx, Y):
    # Create a copy of the dataframe to avoid modifying the original
    df = df.copy()
    
    # Combine X and T for risk model
    X_risk = pd.concat([df[X_columns], df[[Tx]]], axis=1)
    
    # Fit logistic regression for risk
    risk_model = LogisticRegression(max_iter=1000)
    risk_model.fit(X_risk, df[Y])
    
    # Predict risk (uplift??)
    df['risk'] = risk_model.predict_proba(X_risk)[:, 1]
    
    # Calculate propensity scores
    prop_model = LogisticRegression(max_iter=1000)
    prop_model.fit(df[X_columns], df[Tx])
    df['propensity'] = prop_model.predict_proba(df[X_columns])[:, 1]

    return df

def compute_weights_training(train_df, risk_col, cost_col, tx_col, k_values, print=0):
    # Calculate average risk and cost for Tx=1 group
    tx_group = train_df[train_df[tx_col] == 1]
    avg_risk_tx = tx_group[risk_col].mean()
    avg_cost_tx = tx_group[cost_col].mean()
    
    # Calculate average risk and cost for Tx=0 group
    no_tx_group = train_df[train_df[tx_col] == 0]
    avg_risk_no_tx = no_tx_group[risk_col].mean()
    avg_cost_no_tx = no_tx_group[cost_col].mean()
    
    results = []
    
    # Loop over k values
    for k in k_values:
        weights_tx = (1 / (avg_risk_tx ** k)) * (1 / (avg_cost_tx ** (1 - k)))
        weights_no_tx = (1 / (avg_risk_no_tx ** k)) * (1 / (avg_cost_no_tx ** (1 - k)))
        
        results.append({
            'k': k,
            'weights_tx': weights_tx,
            'weights_no_tx': weights_no_tx
        })

    # Convert the results list to a DataFrame for easier plotting
    results_df = pd.DataFrame(results)

    # Plotting the results
    if print:
        plt.figure(figsize=(10, 6))
        plt.plot(results_df['k'], results_df['weights_tx'], marker='o', label='Avg Weights- Tx')
        plt.plot(results_df['k'], results_df['weights_no_tx'], marker='o', label='Avg Weights- No Tx')
        plt.xlabel('k Value')
        plt.ylabel('Weight Value')
        plt.title('Weight Comparison Across k Values')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return pd.DataFrame(results)
    
################################################################
### MODEL 
################################################################

def calculate_owl_weights(df, Tx, k=0.5, alpha=1.0, epsilon=1e-8):
    # Create a copy of the dataframe to avoid modifying the original
    df = df.copy()
    
    # Transform risk (1/R)
    df['transformed_risk'] = 1 / (df['risk'] + epsilon)
    
    # Normalize and amplify transformed risk 
    df['norm_trans_risk'] = (df['transformed_risk'] - df['transformed_risk'].min()) / \
                            (df['transformed_risk'].max() - df['transformed_risk'].min())
    df['amplified_risk'] = (df['transformed_risk'] ** (1 + alpha * df['norm_trans_risk']))**k

    # Normalize and amplify cost
    df['norm_trans_cost'] = (df['cost'] - df['cost'].min()) / (df['cost'].max() - df['cost'].min())
    df['amplified_cost'] = (df['cost'] ** (1 - alpha * df['norm_trans_cost']))**(1-k)

    # Calculate weights (R^k/C^1-k/propensity)
    df['weight'] = np.where(df[Tx] == 1, 
                            (df['amplified_risk'] / df['amplified_cost']) / df['propensity'],
                            (df['amplified_risk'] / df['amplified_cost']) / (1 - df['propensity']))
     
    return df

def train_owl_svm(train_df, test_df, features, Tx, Y, k=0.5, alpha=1.0, kernel='linear', svm_C=1.0, max_iter=5000, epsilon=1e-8):
    print(f"Starting train_owl_svm with k={k}, SVM C={svm_C}, alpha={alpha}")
    
    # Copy dataframes to avoid modifying the original data
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    # Calculate weights for training and testing data
    weighted_train_df = calculate_owl_weights(train_df, Tx, k, alpha)
    #weighted_test_df = calculate_owl_weights(test_df, k, alpha) 
    
    # Prepare training data
    X_train = weighted_train_df[features]
    y_train = weighted_train_df[Tx]
    weights = weighted_train_df['weight']

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train SVM model
    svm_model = SVC(kernel=kernel, C=svm_C, probability=True, random_state=42, max_iter=max_iter)
    svm_model.fit(X_train_scaled, y_train, sample_weight=weights)

    # Prepare testing data
    X_test = test_df[features]
    X_test_scaled = scaler.transform(X_test)
    test_df['pred_tx'] = svm_model.predict(X_test_scaled)
    
    # Compute metrics
    num_treated_pred = test_df['pred_tx'].sum()

    if num_treated_pred == 0:
        subgroup0_perc, subgroup1_perc, subgroup2_perc = 0, 0, 0
    else:
        subgroup0_perc = test_df.loc[test_df['subgroup'] == 0, 'pred_tx'].sum()/num_treated_pred
        subgroup1_perc = test_df.loc[test_df['subgroup'] == 1, 'pred_tx'].sum()/num_treated_pred
        subgroup2_perc = test_df.loc[test_df['subgroup'] == 2, 'pred_tx'].sum()/num_treated_pred
        
    treated_mask = test_df['pred_tx'] == 1
    total_risk = test_df.loc[treated_mask, 'risk'].sum()
    total_cost = create_cost(test_df, 'pred_tx').loc[treated_mask, 'cost'].sum()
    # Loop through each subgroup
    
    # Initialize dictionaries to store results for each subgroup
    total_risk_sbgrp = {0: 0, 1: 0, 2: 0}
    total_cost_sbgrp = {0: 0, 1: 0, 2: 0}

    for sbgrp in [0, 1, 2]:
        # Create a mask for the current subgroup
        sbgrp_mask = test_df['subgroup'] == sbgrp
        
        # Calculate metrics for the current subgroup
        sbgrp_treated_mask = treated_mask & sbgrp_mask
        total_risk_sbgrp[sbgrp] = test_df.loc[sbgrp_treated_mask, 'risk'].sum()
        total_cost_sbgrp[sbgrp] = create_cost(test_df, 'pred_tx').loc[sbgrp_treated_mask, 'cost'].sum()

    
    test_accuracy = accuracy_score(test_df[Tx], test_df['pred_tx'])

    # Print results
    #print(f"Test set size: {len(test_df)}")
    print(f"Predicted treated: {num_treated_pred} ({num_treated_pred/len(test_df)*100:.2f}%)")
    print(f"Total risk: {total_risk:.2f}")
    print(f"Total Cost: {total_cost:.2f}")
    #print(f"Test accuracy: {test_accuracy:.4f}")
    print("----")
    
    return total_risk, total_cost, num_treated_pred, subgroup0_perc, subgroup1_perc, subgroup2_perc, total_risk_sbgrp, total_cost_sbgrp

def analyze_k_values(train_df, test_df, features, Tx, Y, k_values=0.5, alpha=1.0, svm_C=1.0, max_iter=5000):
    results = []

    # Check if k_values is a single value or a list
    if isinstance(k_values, (int, float)):
        k_values = [k_values]  # Convert single value to a list

    for k in k_values:
        total_inverse_R, total_cost, num_treated, perc_subgroup0, perc_subgroup1, perc_subgroup2, total_risk_sbgrp, total_cost_sbgrp = train_owl_svm(train_df, test_df, features, Tx, Y, k=k, alpha=alpha, svm_C=svm_C, max_iter=max_iter)
        results.append({
            'k': k, 
            #'test_accuracy': test_accuracy, 
            'total_risk': total_inverse_R, 
            'total_cost': total_cost, 
            'num_treated': num_treated,
            'perc_subgroup_0': perc_subgroup0,
            'perc_subgroup_1': perc_subgroup1,
            'perc_subgroup_2': perc_subgroup2,
            'total_risk_0':  total_risk_sbgrp[0],
            'total_risk_1':  total_risk_sbgrp[1],
            'total_risk_2':  total_risk_sbgrp[2],
            'total_cost_0':  total_cost_sbgrp[0],
            'total_cost_1':  total_cost_sbgrp[1],
            'total_cost_2':  total_cost_sbgrp[2]
        })
    return pd.DataFrame(results)

def plot_R_vs_C_with_treatment(results_df):
    
    # 1. Plot total_risk vs total_cost
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['total_cost'], results_df['total_risk'], c=results_df['k'], cmap='viridis', edgecolors='k')
    plt.colorbar(label='k value')
    plt.xlabel('Total Cost (C)')
    plt.ylabel('Total Risk (R)')
    plt.title('Total Risk (R) vs. Total Cost (C) for Different k Values')
    plt.grid(True)
    plt.show()

    # 2. Plot the number of treated patients against k values
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['k'], results_df['num_treated'], marker='o', linestyle='-', color='b')
    plt.xlabel('k value')
    plt.ylabel('Number of Treated Patients')
    plt.title('Number of Treated Patients vs. k Value')
    plt.grid(True)
    plt.show()

    # 3. Plot each subgroup
    plt.plot(results_df['k'], results_df['perc_subgroup_0'], marker='o', linestyle='-', color='r', label='Subgroup 0')
    plt.plot(results_df['k'], results_df['perc_subgroup_1'], marker='o', linestyle='-', color='g', label='Subgroup 1')
    plt.plot(results_df['k'], results_df['perc_subgroup_2'], marker='o', linestyle='-', color='b', label='Subgroup 2')
    
    # Add labels and title
    plt.xlabel('k value')
    plt.ylabel('Number of Patients')
    plt.title('Percentage of Patients in Subgroups by k Value')
    plt.grid(True)
    
    # Add legend
    plt.legend()
    
    # Show the plot
    plt.show()

    # 4. Plot each subgroup's total_risk
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['k'], results_df['total_risk_0'], marker='o', linestyle='-', color='r', label='Subgroup 0')
    plt.plot(results_df['k'], results_df['total_risk_1'], marker='o', linestyle='-', color='g', label='Subgroup 1')
    plt.plot(results_df['k'], results_df['total_risk_2'], marker='o', linestyle='-', color='b', label='Subgroup 2')
    
    # Add labels and title
    plt.xlabel('k value')
    plt.ylabel('Total Risk')
    plt.title('Total Risk by Subgroup and k Value')
    plt.grid(True)
    
    # Add legend
    plt.legend()
    
    # Show the plot
    plt.show()

    # 5. Plot each subgroup's total cost
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['k'], results_df['total_cost_0'], marker='o', linestyle='-', color='r', label='Subgroup 0')
    plt.plot(results_df['k'], results_df['total_cost_1'], marker='o', linestyle='-', color='g', label='Subgroup 1')
    plt.plot(results_df['k'], results_df['total_cost_2'], marker='o', linestyle='-', color='b', label='Subgroup 2')
    
    # Add labels and title
    plt.xlabel('k value')
    plt.ylabel('Total Cost')
    plt.title('Total Cost by Subgroup and k Value')
    plt.grid(True)
    
    # Add legend
    plt.legend()
    
    # Show the plot
    plt.show()

