import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

######################

def sigmoid(x, L, k, x0):
        """Generalized sigmoid function"""
        return L / (1 + np.exp(-k * (x - x0)))

def T1_effectiveness(x, L=1.5, k=1, x0=5):
    """S-shaped effectiveness curve for T1"""
    return sigmoid(x, L, k, x0)

def T2_effectiveness(x, L=2.0, k=0.8, x0=7):
    """S-shaped effectiveness curve for T2"""
    return sigmoid(x, L, k, x0)
    
######################

def R0(x, a, b, c):
    return a * x + (b - c)
        
def R11(R0_values, treatment):
    if treatment == 'T1':
        percentiles = np.percentile(R0_values, [60, 80])
        multipliers = np.where(R0_values <= percentiles[0], 1.2,
                               np.where(R0_values <= percentiles[1], 1.5, 1.3))
    elif treatment == 'T2':
        percentiles = np.percentile(R0_values, [30, 70])
        multipliers = np.where(R0_values <= percentiles[0], 1.3,
                               np.where(R0_values <= percentiles[1], 1.9, 1.9))
    return R0_values * multipliers

def R1(R0_values, treatment):
    if treatment == 'T1':
        percentiles = np.percentile(R0_values, [60, 80])
        multipliers = np.where(R0_values <= percentiles[0], 1.2,
                               np.where(R0_values <= percentiles[1], 1.5, 1.3))
    elif treatment == 'T2':
        percentiles = np.percentile(R0_values, [30, 70])
        multipliers = np.where(R0_values <= percentiles[0], 1.3,
                               np.where(R0_values <= percentiles[1], 1.9, 1.9))
    return R0_values * multipliers

def C1(R0_values, treatment):
    if treatment == 'T1':
        percentiles = np.percentile(R0_values, [60, 80])
        return np.where(R0_values <= percentiles[0], 2, np.where(R0_values <= percentiles[1], 3, 4))
    
    elif treatment == 'T2':
        percentiles = np.percentile(R0_values, [30, 70])
        return np.where(R0_values <= percentiles[0], 4, np.where(R0_values <= percentiles[1], 1, 2)) 

def R1(R0_values, treatment):
    if treatment == 'T1':
        percentiles = np.percentile(R0_values, [60, 80])
        multipliers = np.where(R0_values <= percentiles[0], 1.15,  # Slightly increase T1 effectiveness for low R0
                               np.where(R0_values <= percentiles[1], 1.25,  # Adjust mid R0 effectiveness
                                        1.05))  # Keep high R0 effectiveness moderate
    elif treatment == 'T2':
        percentiles = np.percentile(R0_values, [30, 70])
        multipliers = np.where(R0_values <= percentiles[0], 1.05,  # Reduce T2 effectiveness for low R0
                               np.where(R0_values <= percentiles[1], 2.3,  # Slightly reduce mid R0 effectiveness
                                        2.7))  # Keep high R0 effectiveness strong, but not overwhelming
    return R0_values * multipliers

def C1(R0_values, treatment):
    if treatment == 'T1':
        percentiles = np.percentile(R0_values, [60, 80])
        return np.where(R0_values <= percentiles[0], 2,  # Low cost for low R0
                        np.where(R0_values <= percentiles[1], 4,  # Moderate cost for mid R0
                                 7))  # Lower high R0 cost to keep T1 competitive
    elif treatment == 'T2':
        percentiles = np.percentile(R0_values, [30, 70])
        return np.where(R0_values <= percentiles[0], 1,  # Keep T2's cost low for low R0
                        np.where(R0_values <= percentiles[1], 4,  # Keep mid R0 cost moderate
                                 5))  # Keep high R0 cost competitive but moderate

######################

def amplified_RC_separate(RC_values, k, alpha, epsilon):
    RC_min, RC_max = np.min(RC_values), np.max(RC_values)
    exp =  1 + alpha * ((RC_values - RC_min) / (RC_max - RC_min + epsilon))
    return RC_values ** exp

def ranked_ratios_amplified_separate(R_values, k, C_values, alpha, epsilon=1e-6):
    RC = (R_values**k / C_values**(1 - k))
    return amplified_RC_separate(RC, k, alpha, epsilon)

######################

def relative_amplification(R_values, k, C_values, alpha, epsilon=1e-6):
    RC = (R_values**k / C_values**(1 - k))
    RC_min, RC_max = np.min(RC), np.max(RC)
    
    # Normalize RC values
    RC_norm = (RC - RC_min) / (RC_max - RC_min + epsilon)
    
    # Apply amplification to normalized values
    RC_amp = RC_norm * (1 + alpha * RC_norm)
    
    # Re-scale back to original range
    return RC_min + (RC_max - RC_min) * RC_amp

def evaluate_treatments(x, k_val, alpha, C0, a, b, c):
    R0_values = R0(x, a, b, c)
    
    # T1_effect = T1_effectiveness(x)
    # T2_effect = T2_effectiveness(x)
    T1_effect = R1(R0_values, 'T1')
    T2_effect = R1(R0_values, 'T2')
    
    R1_T1 = R0_values * T1_effect
    R1_T2 = R0_values * T2_effect
    
    C1_T1 = C1(R0_values, 'T1')
    C1_T2 = C1(R0_values, 'T2')

    amplified_R0 = ranked_ratios_amplified_separate(R0_values, k_val, C0 * np.ones_like(R0_values), alpha)
    amplified_R1_T1 = ranked_ratios_amplified_separate(R1_T1, k_val, C1_T1, alpha)
    amplified_R1_T2 = ranked_ratios_amplified_separate(R1_T2, k_val, C1_T2, alpha)

    treatment_choice = np.argmax([amplified_R0, amplified_R1_T1, amplified_R1_T2], axis=0)
    
    # Calculate additional metrics
    avg_R0_T1 = np.mean(R0_values[treatment_choice == 1]) if np.any(treatment_choice == 1) else 0
    avg_R0_T2 = np.mean(R0_values[treatment_choice == 2]) if np.any(treatment_choice == 2) else 0
    avg_R0_combined = np.mean(R0_values[treatment_choice != 0]) if np.any(treatment_choice != 0) else 0
    
    total_R = np.sum(np.where(treatment_choice == 0, R0_values,
                              np.where(treatment_choice == 1, R1_T1, R1_T2)))
    total_C = np.sum(np.where(treatment_choice == 0, C0,
                              np.where(treatment_choice == 1, C1_T1, C1_T2)))
    
    return treatment_choice, avg_R0_T1, avg_R0_T2, avg_R0_combined, total_R, total_C

######################

def Plotting(x, summary_df, decisions, a, b, c):
    R0_values = R0(x, a, b, c)
    
    # Plot Treatment effectiveness
    plt.figure(figsize=(12, 8))
    plt.plot(x, R1(R0_values, 'T1'), label='T1 Effectiveness') 
    plt.plot(x, R1(R0_values, 'T2'), label='T2 Effectiveness')
    plt.plot(x, R0_values, label='R0', linestyle='--') #np.max(R0(x))
    plt.xlabel('x')
    plt.ylabel('Effectiveness')
    plt.title('Effectiveness of T1 and T2 Treatments')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot Number Treated vs k
    plt.figure(figsize=(12, 6))
    plt.plot(summary_df['k'], summary_df['Num Tx T1'], marker='o', label='T1')
    plt.plot(summary_df['k'], summary_df['Num Tx T2'], marker='o', label='T2')
    plt.plot(summary_df['k'], summary_df['Num Tx Total'], marker='o', label='Total Treated')
    plt.xlabel('k')
    plt.ylabel('Number Treated')
    plt.title('Number Treated vs k (Amplified, Alpha=1.5)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot Total R vs Total C
    plt.figure(figsize=(10, 6))
    plt.scatter(summary_df['Total C'], summary_df['Total R'], c=summary_df['k'], cmap='viridis')
    plt.colorbar(label='k value')
    plt.xlabel('Total Cost (C)')
    plt.ylabel('Total Outcome (R)')
    plt.title('Total R vs Total C across k spectrum (Optimal Treatment Choice)')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.imshow(decisions, extent=[0, 10, 0, 1], origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(ticks=[0, 1, 2], label='Treatment Decision')
    plt.clim(-0.5, 2.5)
    plt.xlabel('x (patient characteristic / severity)')
    plt.ylabel('k (outcome vs. cost weight)')
    plt.title('Optimal Treatment Decision based on x and k')
    plt.show()

######################
# Main analysis

def main_prog(a, b, c, C0, epsilon, alpha):
    x_range = np.linspace(0, 10, 100)
    k_range = np.linspace(0, 1, 100)  
    X, K = np.meshgrid(x_range, k_range)
    
    # Calculate decisions for both heatmap and summary
    # decisions = np.array([evaluate_treatments(x, k, alpha, C0, a, b, c)[0] for x, k in zip(X.ravel(), K.ravel())]).reshape(X.shape)

    decisions = []  # To hold the treatment decisions for each k
    
    for k_val in k_range:  # For each k
        # Evaluate treatments for the full range of x at once for this k
        treatment_choice, _, _, _, _, _ = evaluate_treatments(x_range, k_val, alpha, C0, a, b, c)
        decisions.append(treatment_choice)  # Append the treatment decisions for this k
    
    decisions = np.array(decisions) 
    
    # Generate summary data
    summary_data = []
    summary_data = []
    for k_val in np.linspace(0, 1, 11):  # Keep 11 points for summary for readability
        treatment_choice = decisions[np.argmin(np.abs(k_range - k_val)), :]
        num_T1 = np.sum(treatment_choice == 1)
        num_T2 = np.sum(treatment_choice == 2)
        avg_R0_T1 = np.mean(R0(x_range, a, b, c)[treatment_choice == 1]) if num_T1 > 0 else 0
        avg_R0_T2 = np.mean(R0(x_range, a, b, c)[treatment_choice == 2]) if num_T2 > 0 else 0
        #avg_R0_combined = np.mean(R0(x_range)[treatment_choice != 0]) if (num_T1 + num_T2) > 0 else 0
        avg_R0_no_tx = np.mean(R0(x_range, a, b, c)[treatment_choice == 0]) if np.sum(treatment_choice == 0) > 0 else 0
    
        total_R = np.sum(np.where(treatment_choice == 0, R0(x_range, a, b, c),
                                  np.where(treatment_choice == 1, R0(x_range, a, b, c) * R1(R0(x_range, a, b, c), 'T1'),
                                           R0(x_range, a, b, c) * R1(R0(x_range, a, b, c), 'T2'))))
        total_C = np.sum(np.where(treatment_choice == 0, C0,
                                  np.where(treatment_choice == 1, C1(R0(x_range, a, b, c), 'T1'),
                                           C1(R0(x_range, a, b, c), 'T2'))))
        summary_data.append({
            'k': k_val,
            'Num Tx T1': num_T1,
            'Num Tx T2': num_T2,
            'Num Tx Total': num_T1 + num_T2,
            'Avg R0 No Tx': avg_R0_no_tx,
            'Avg R0 T1': avg_R0_T1,
            'Avg R0 T2': avg_R0_T2,
            #'Avg R0 Combined': avg_R0_combined,
            'Total R': total_R,
            'Total C': total_C
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Plotting
    Plotting(x_range, summary_df, decisions, a, b, c)
    
    print("Summary of Optimal Treatment Choices:")
    print(summary_df.to_string(index=False, float_format='{:,.2f}'.format))

