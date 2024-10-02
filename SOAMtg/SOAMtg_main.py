import importlib
import SOAMtg_data as soa_data
importlib.reload(soa_data)
import SOAMtg_model as soa_mdl
importlib.reload(soa_mdl)
import SOAMtg_code as soa_code
importlib.reload(soa_code)
from sklearn.model_selection import train_test_split

# Constants
a, b, c = 2, 5, 3
C0, epsilon = 1, 1e-6
alpha = 1.5

###################
percentiles_R = {
    'T1': [70, 90],  # High return cases for T1
    'T2': [40, 80]   # Broader range for T2
}
percentiles_C = {
    'T1': [70, 90],  # High cost cases for T1
    'T2': [50, 90]   # Mid to high cost cases for T2
}
impact_factors = {
    'T1': [0.5, 0.3, 0.1],  # High impact for T1 in top percentiles
    'T2': [0.2, 0.5, 0.2]   # Mixed impact for T2
}
cost_factors = {
    'T1': [3, 6, 9],   # Expensive treatments for T1
    'T2': [2, 4, 8]    # Moderate to expensive treatments for T2
}

###################
percentiles_R = {
    'T1': [60, 85],  # T1 is more effective for higher-risk patients
    'T2': [40, 75]   # T2 has a broader impact range
}

percentiles_C = {
    'T1': [60, 85],  # T1 is more effective for higher-risk patients
    'T2': [40, 75]   # T2 has a broader impact range
}

# Define new impact_factors
impact_factors = {
    'T1': [0.05, 0.15, 0.25],  # T1 has higher impact on highest risk group
    'T2': [0.10, 0.20, 0.15]   # T2 has highest impact on middle risk group
}

# Define cost factors (new addition to create cost variation)
cost_factors = {
    'T1': [800, 1200, 1600],   # T1 costs increase with risk
    'T2': [1500, 2000, 1800]   # T2 has high initial cost, slightly lower for highest risk
}


###################
# 1. DATA/ACTUAL
data = soa_data.get_data('Readmit_R.csv')
df = soa_data.create_risk(data)
df1 = soa_data.create_treatment(df, p1_treat=0.2, p2_treat=0.2, percentiles_R=percentiles_R,impact_factors=impact_factors)
treat_summary = soa_data.summarize_treatment_allocation(df1)
print(treat_summary)

features_df = df1.drop(['Readmit','Readmit_red','ID','preds'],axis=1)

A = df1.columns.to_list()
B = ['Readmit','Readmit_red','ID','preds','Tx']
features = list(set(A) - set(B)) 

train_index, test_index = train_test_split(df1.index, test_size=0.3, random_state=0)    
train_df = df1.iloc[train_index]
test_df  = df1.iloc[test_index]

# 2. PREDICT (RE-TRAIN/PREDICT)
learner_method = 's' #'t'
learner_results = soa_mdl.learner(train_df, test_df, features, 'Readmit_red', 'Tx', method=learner_method)

# 3. CLASSIFTY
soa_code.main_prog(train_df, test_df, features, 'Readmit_red', 'Tx', 
          C0, epsilon, alpha, cost_factors, percentiles_C, learner_results)

