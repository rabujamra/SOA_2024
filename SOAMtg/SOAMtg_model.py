import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def preds(train_df, test_df, features, outcomeCol, treatmentCol, no_of_splits_kfold = 5):

    propModel = GradientBoostingClassifier().fit(train_df[features], train_df[outcomeCol])
    preds = propModel.predict_proba(test_df[features])[:,1]

    return preds

def s_learner(train_df, test_df, features, outcomeCol, treatmentCol):
    # Get unique treatment values
    unique_treatments = sorted(train_df[treatmentCol].unique())
    
    # Make an explicit copy of test_df to avoid SettingWithCopyWarning
    test_df = test_df.copy()
    
    # Initialize result dictionary for multiple treatments
    results = {}
    
    # Create a modified treatment column for the entire dataset once
    train_df[f'{treatmentCol}_mod'] = train_df[treatmentCol]
    test_df[f'{treatmentCol}_mod'] = test_df[treatmentCol]
    
    # Train model once
    propModel = GradientBoostingClassifier(random_state=42)  # Add random_state for reproducibility
    propModel.fit(train_df[features + [f'{treatmentCol}_mod']], train_df[outcomeCol])
    
    # Loop over treatments or handle single treatment case
    for treatment_val in unique_treatments[1:]:  # Skip control (0)
        # Predict for current treatment
        test_df[f'{treatmentCol}_mod'] = treatment_val
        treat_data_val_pred = propModel.predict_proba(test_df[features + [f'{treatmentCol}_mod']])[:, 1]
        
        # Predict for control
        test_df[f'{treatmentCol}_mod'] = 0
        control_data_val_pred = propModel.predict_proba(test_df[features + [f'{treatmentCol}_mod']])[:, 1]
        
        # Calculate treatment effect
        treatment_effect = treat_data_val_pred - control_data_val_pred
        
        # Store results
        results[f'Tx_{treatment_val}_vs_Control'] = {
            'treatment_pred': treat_data_val_pred,
            'control_pred': control_data_val_pred,
            'treatment_effect': treatment_effect
        }
    
    return results if len(unique_treatments) > 2 else results[f'Tx_{unique_treatments[1]}_vs_Control']

def t_learner(train_df, test_df, features, outcomeCol, treatmentCol):
    unique_treatments = sorted(train_df[treatmentCol].unique())
    test_df = test_df.copy()
    
    results = {}
    control_model = GradientBoostingClassifier(random_state=42)
    control_data = train_df[train_df[treatmentCol] == 0]
    control_model.fit(control_data[features], control_data[outcomeCol])
    
    for treatment_val in unique_treatments[1:]:  # Skip control (0)
        treatment_model = GradientBoostingClassifier(random_state=42)
        treatment_data = train_df[train_df[treatmentCol] == treatment_val]
        treatment_model.fit(treatment_data[features], treatment_data[outcomeCol])
        
        treat_data_val_pred = treatment_model.predict_proba(test_df[features])[:, 1]
        control_data_val_pred = control_model.predict_proba(test_df[features])[:, 1]
        
        treatment_effect = treat_data_val_pred - control_data_val_pred
        
        results[f'Tx_{treatment_val}_vs_Control'] = {
            'treatment_pred': treat_data_val_pred,
            'control_pred': control_data_val_pred,
            'treatment_effect': treatment_effect
        }
    
    return results

def learner(train_df, test_df, features, outcomeCol, treatmentCol, method='s'):
    if method.lower() == 's':
        return s_learner(train_df, test_df, features, outcomeCol, treatmentCol)
    elif method.lower() == 't':
        return t_learner(train_df, test_df, features, outcomeCol, treatmentCol)
    else:
        raise ValueError("Method must be 's' for S-learner or 't' for T-learner")
