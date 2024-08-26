import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

import OWL_Funcs as func

###########################################################################

def generate_X_A(n, n_col, tx):
    X = np.random.uniform(low=-1, high=1, size=(n, n_col))
    if tx==2:
        A = np.random.binomial(n=1, p=0.5, size = n)
    else: 
        A = np.random.randint(3, size = n) 
    return X, A

def generate_positive_R_k(R_k, X, A, c, num_positive_values=0):
    positive_values = []
    while len(positive_values) < num_positive_values:
        R = R_k(X, A, c)
        R_positive = R[R > 0]
        positive_values.extend(R_positive)
        if len(positive_values) < num_positive_values:
            additional_samples = num_positive_values - len(positive_values)
            X, A = generate_X_A(additional_samples, X.shape[1], len(np.unique(A)))
    positive_values = positive_values[:num_positive_values]
    return np.array(positive_values)
    
def owl_initial(cons, R_k, tx=2, n_train=800, n_test=1000):
    n_col = 50
    X_train, T = generate_X_A(n_train, n_col, tx)
    X_test, _ = generate_X_A(n_test, n_col, tx)
    #R = generate_positive_R_k(R_k, X_train, T, cons, n_train)
    R = R_k(X_train, T, cons)
    
    positive_indices = R > 0
    R = R[positive_indices]
    
    # Filter X_train and T to match positive R values
    X_train = X_train[positive_indices]
    T = T[positive_indices]

    C = func.calc_cost(T, len(T))
    
    return X_train, T, R, X_test, C

#####

def generate_positive_R_k1(R_k, X, A, c, num_positive_values):
    positive_values = []
    while len(positive_values) < num_positive_values:
        R = R_k(X, A, c)
        R_positive = R[R > 0]
        positive_values.extend(R_positive)
        if len(positive_values) < num_positive_values:
            additional_samples = num_positive_values - len(positive_values)
            X = np.random.uniform(low=-1, high=1, size=(additional_samples, X.shape[1]))
            A = np.random.choice([0, 1, 2], size=additional_samples) if len(np.unique(A)) == 3 else np.random.binomial(n=1, p=0.5, size=additional_samples)
    positive_values = positive_values[:num_positive_values]
    return np.array(positive_values)
    
def owl_initial1(cons, R_k, tx=2, n_train=800, n_test=1000):
    n_col = 50
    X_train = np.random.uniform(low = -1, high = 1, size = (n_train, n_col))
    if tx==2:
        T = np.random.binomial(n=1, p=0.5, size = n_train)
    else: 
        T = np.random.randint(3, size = n_train) 
    
    X_test = np.random.uniform(low = -1, high = 1, size = (n_test, n_col))
    #R = R_k(X_train,T,cons)
    R = generate_positive_R_k(R_k, X_train,T,cons, n_train)
    
    return T, R, X_train, X_test

def owl_propen(X_train, T):
    n = len(T)
    pi = np.zeros(n)  
    probs = LogisticRegression().fit(X_train, T).predict_proba(X_train)
    
    for t in np.unique(T):
        pi += probs[:, t] * (T == t)  

    epsilon = np.finfo(float).eps
    pi[pi == 0] = epsilon

    return pi

def grid_search_svc(X_train, T, R, pi, kernel='linear', search_type='grid', n_iter=100):
    # Set up common parameters
    scoring = ['accuracy']
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    svc = svm.SVC()

    # Define parameter ranges
    if search_type == 'grid':
        C_range = np.logspace(-2, 2, 5)
        gamma_range = np.logspace(-2, 2, 5)
    elif search_type == 'random':
        C_range = np.logspace(-5, 5, 11)
        gamma_range = np.logspace(-5, 5, 11)
    else:
        raise ValueError("Invalid search_type. Choose 'grid' or 'random'.")

    # Create the parameter grid based on the kernel
    if kernel == 'linear':
        param_grid = {
            "C": C_range,
            "kernel": ['linear']
        }
        if search_type == 'random':
            # Adjust n_iter based on parameter combinations for 'linear'
            n_iter = min(n_iter, len(C_range))
    elif kernel == 'rbf':
        param_grid = {
            "C": C_range,
            "kernel": ['rbf'],
            "gamma": gamma_range.tolist() + ['scale', 'auto']
        }
    else:
        raise ValueError("Invalid kernel. Choose 'linear' or 'rbf'.")

    # Set up search method
    if search_type == 'grid':
        search = GridSearchCV(estimator=svc,
                              param_grid=param_grid,
                              scoring=scoring,
                              refit='accuracy',
                              n_jobs=-1,
                              cv=kfold,
                              verbose=0)
    elif search_type == 'random':
        search = RandomizedSearchCV(estimator=svc,
                                    param_distributions=param_grid,
                                    n_iter=n_iter,
                                    scoring=scoring,
                                    refit='accuracy',
                                    n_jobs=-1,
                                    cv=kfold,
                                    verbose=0)
    else:
        raise ValueError("Invalid search_type. Choose 'grid' or 'random'.")

    # Fit search
    result = search.fit(X_train, T, sample_weight=R / pi)

    return result

def calculate_value_function(X_test, predicted_treatments, R_k, cons):
    # Generate simulated rewards based on X_test and predicted_treatments
    simulated_rewards = R_k(X_test, predicted_treatments, cons)
    
    # Calculate the average of these simulated rewards
    value_function = np.mean(simulated_rewards)
    print(f"Avg Value Function: {value_function}")
    
def evaluate_model(prediction, T_actual):
    accuracy = accuracy_score(T_actual, prediction)
    precision = precision_score(T_actual, prediction, average='macro')
    conf_matrix = confusion_matrix(T_actual, prediction)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Conf Matrix: {conf_matrix}")
    
def owl_plot(X_test, prediction, optimal_k):
    # Scatter plot of the predictions
    plt.figure(figsize=(8, 6))
    scatter = sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=prediction, palette='viridis', edgecolor='k')
    
    # Create a grid of values
    x = np.linspace(X_test[:, 0].min() - 0.1, X_test[:, 0].max() + 0.1, 400)
    y = np.linspace(X_test[:, 1].min() - 0.1, X_test[:, 1].max() + 0.1, 400)
    X, Y = np.meshgrid(x, y)
    
    # Compute the function values
    Z = optimal_k(X, Y)
    
    # Add contour lines to delineate regions
    plt.contour(X, Y, Z, levels=np.array([0, 1]), colors='maroon', linestyles='solid')

    # Add labels and title
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('OWL Predictions by Treatments (T1, T2, T3)')
    plt.grid(True)

    # Create a legend with custom labels
    handles, labels = scatter.get_legend_handles_labels()
    custom_labels = ['T1', 'T2', 'T3']  # Modify these labels as needed
    plt.legend(handles=handles, labels=custom_labels, title='Prediction', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Show plot
    plt.show()

def plot_function(optimal_k):
    # Create a grid of values
    x = np.linspace(-1, 1, 400)
    y = np.linspace(-1, 1, 400)
    X, Y = np.meshgrid(x, y)
    
    plt.figure(figsize=(8, 6))

    Z = optimal_k(X, Y)
    print(f"Shape of Z: {Z.shape}")
    
    # Add contour lines to delineate regions
    contour_lines = plt.contour(X, Y, Z, levels=np.array([0, 1]), colors='k', linestyles='solid')

    # Add labels and title
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.grid(True)

    # Show plot
    plt.show()
