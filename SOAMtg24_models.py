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

################################################################
### MODEL 
################################################################

def calculate_owl_weights(df, X_columns, Tx, cost, Y, k=0.5, alpha=1.0, epsilon=1e-8):
    # Create a copy of the dataframe to avoid modifying the original
    df = df.copy()
    
    # Combine X and T for risk model
    X_risk = pd.concat([df[X_columns], df[[Tx]]], axis=1)
    
    # Fit logistic regression for risk
    risk_model = LogisticRegression(max_iter=1000)
    risk_model.fit(X_risk, df[Y])
    
    # Predict risk
    df['risk'] = risk_model.predict_proba(X_risk)[:, 1]
    
    # Calculate propensity scores
    prop_model = LogisticRegression(max_iter=1000)
    prop_model.fit(df[X_columns], df[Tx])
    df['propensity'] = prop_model.predict_proba(df[X_columns])[:, 1]
    
    # Transform risk (1/R)
    df['transformed_risk'] = 1 / (df['risk'] + epsilon)
    
    # Normalize and amplify transformed risk 
    df['norm_trans_risk'] = (df['transformed_risk'] - df['transformed_risk'].min()) / \
                            (df['transformed_risk'].max() - df['transformed_risk'].min())
    df['amplified_risk'] = (df['transformed_risk'] ** (1 + alpha * df['norm_trans_risk']))**k

    # Normalize and amplify cost
    df['norm_trans_cost'] = (df[cost] - df[cost].min()) / (df[cost].max() - df[cost].min())
    df['amplified_cost'] = (df[cost] ** (1 - alpha * df['norm_trans_cost']))**(1-k)

    # Calculate weights (R^k/C^1-k/propensity)
    df['weight'] = np.where(df[Tx] == 1, 
                            (df['amplified_risk'] / df['amplified_cost']) / df['propensity'],
                            (df['amplified_risk'] / df['amplified_cost']) / (1 - df['propensity']))
     
    return df

def train_owl_svm(train_df, test_df, features, Tx, cost_col, Y, k=0.5, alpha=0.2, kernel='linear', svm_C=1.0, max_iter=5000, epsilon=1e-8):
    print(f"Starting train_owl_svm with k={k}, SVM C={svm_C}")
    
    # Copy dataframes to avoid modifying the original data
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    # Calculate weights for training and testing data
    weighted_train_df = calculate_owl_weights(train_df, features, Tx, cost_col, Y, k, alpha)
    weighted_test_df = calculate_owl_weights(test_df, features, Tx, cost_col, Y, k, alpha) 
    
    # Prepare training data
    X_train = weighted_train_df[features]
    y_train = weighted_train_df[Tx]
    weights = weighted_train_df['weight']

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train SVM model
    #treated_mask1 = weighted_test_df['Tx'] == 1
    #print(f"transformed risk Tx: {weighted_test_df.loc[treated_mask1, 'transformed_risk'].mean()}")
    #print(f"transformed risk ~Tx: {weighted_test_df.loc[~treated_mask1, 'transformed_risk'].mean()}")
    #print(f"cost Tx: {weighted_test_df.loc[treated_mask1, 'cost'].mean()}")
    #print(f"cost ~Tx: {weighted_test_df.loc[~treated_mask1, 'cost'].mean()}")
    #print(f"weight Tx: {weighted_test_df.loc[treated_mask1, 'weight'].mean()}")
    #print(f"weight ~Tx: {weighted_test_df.loc[~treated_mask1, 'weight'].mean()}")
    #print(weighted_train_df.head())
    #svm_model = SVC(kernel=kernel, C=svm_C, probability=True, random_state=42)
    svm_model = SVC(kernel=kernel, C=svm_C, probability=True, random_state=42, max_iter=max_iter)
    svm_model.fit(X_train_scaled, y_train, sample_weight=weights)

    # Prepare testing data
    X_test = weighted_test_df[features]
    X_test_scaled = scaler.transform(X_test)
    weighted_test_df['pred_tx'] = svm_model.predict(X_test_scaled)
    
    # Compute metrics
    num_treated_pred = weighted_test_df['pred_tx'].sum()
    
    treated_mask = weighted_test_df['pred_tx'] == 1
    total_inverse_R = weighted_test_df.loc[treated_mask, 'transformed_risk'].sum()
    total_cost = create_cost(weighted_test_df, 'pred_tx').loc[treated_mask, 'cost'].sum()

    #print(weighted_test_df.head())
    
    test_accuracy = accuracy_score(test_df[Tx], weighted_test_df['pred_tx'])

    # Print results
    #print(f"Test set size: {len(weighted_test_df)}")
    print(f"Predicted treated: {num_treated_pred} ({num_treated_pred/len(weighted_test_df)*100:.2f}%)")
    print(f"Total Inverse R: {total_inverse_R:.2f}")
    print(f"Total Cost: {total_cost:.2f}")
    #print(f"Test accuracy: {test_accuracy:.4f}")
    print("----")
    
    return test_accuracy, total_inverse_R, total_cost, num_treated_pred

def analyze_k_values(train_df, test_df, features, Tx, cost_col, Y, k_values=np.arange(0, 1.1, 0.1), svm_C=1.0, max_iter=5000):
    results = []
    for k in k_values:
        test_accuracy, total_inverse_R, total_cost, num_treated = train_owl_svm(
            train_df, test_df, features, Tx, cost_col, Y, k=k, svm_C=svm_C, max_iter=max_iter
        )
        results.append({
            'k': k, 
            'test_accuracy': test_accuracy, 
            'total_inverse_R': total_inverse_R, 
            'total_cost': total_cost, 
            'num_treated': num_treated
        })
    return pd.DataFrame(results)

def plot_R_vs_C_with_treatment(results_df):
    # Plot total_inverse_R vs total_cost
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['total_cost'], results_df['total_inverse_R'], c=results_df['k'], cmap='viridis', edgecolors='k')
    plt.colorbar(label='k value')
    plt.xlabel('Total Cost (C)')
    plt.ylabel('Total Inverse Reward (R)')
    plt.title('Total Inverse Reward (R) vs. Total Cost (C) for Different k Values')
    plt.grid(True)
    plt.show()

    # Plot the number of treated patients against k values
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['k'], results_df['num_treated'], marker='o', linestyle='-', color='b')
    plt.xlabel('k value')
    plt.ylabel('Number of Treated Patients')
    plt.title('Number of Treated Patients vs. k Value')
    plt.grid(True)
    plt.show()
