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

#############

# Cost function
def C1(R0_values, treatment, percentiles_C, cost_factors):
    p1, p2 = percentiles_C[treatment]
    low_cost, mid_cost, high_cost = cost_factors[treatment]
    cutoffs = np.percentile(R0_values, [p1, p2])
    return np.where(R0_values <= cutoffs[0], low_cost,
                    np.where(R0_values <= cutoffs[1], mid_cost, high_cost))

def C0_adjust(R0_values, C0_base, epsilon=1e-6):
    min_R0 = float(min(R0_values))
    max_R0 = float(max(R0_values))

    # Prevent division by zero with epsilon
    if max_R0 == min_R0:
        max_R0 += epsilon

    # Adjust C0 based on R0
    C0_adjusted = [(C0_base * (R0 - min_R0) / (max_R0 - min_R0)) for R0 in R0_values]
    return C0_adjusted



######################

# 2nd Amplification function
def amplified_RC_separate(RC_values, k, alpha, epsilon):
    RC_min, RC_max = np.min(RC_values), np.max(RC_values)
    exp =  1 + alpha * ((RC_values - RC_min) / (RC_max - RC_min + epsilon))
    return RC_values ** exp

# Amplification function
def ranked_ratios_amplified_separate(R_values, k, C_values, alpha, epsilon=1e-6):
    RC = ((1/R_values)**k / C_values**(1 - k))
    return amplified_RC_separate(RC, k, alpha, epsilon)

######################

# Treatment evaluation function
def evaluate_treatments(learner_results, k_val, alpha, C0_base, cost_factors, percentiles_C):
    R0_values = learner_results['Tx_1_vs_Control']['control_pred']
    R1_values = learner_results['Tx_1_vs_Control']['treatment_pred']
    R2_values = learner_results['Tx_2_vs_Control']['treatment_pred']

    C0 = C0_adjust(R0_values, C0_base)
    C1_T1 = C1(R0_values, 'T1', percentiles_C, cost_factors)
    C1_T2 = C1(R0_values, 'T2', percentiles_C, cost_factors)
    
    amplified_R0 = ranked_ratios_amplified_separate(R0_values, k_val, C0, alpha) #* np.ones_like(R0_values)
    amplified_R1 = ranked_ratios_amplified_separate(R1_values, k_val, C1_T1, alpha)
    amplified_R2 = ranked_ratios_amplified_separate(R2_values, k_val, C1_T2, alpha)
    
    treatment_choice = np.argmax([amplified_R0, amplified_R1, amplified_R2], axis=0)
    
    avg_R0_T1 = np.mean(R0_values[treatment_choice == 1]) if np.any(treatment_choice == 1) else 0
    avg_R0_T2 = np.mean(R0_values[treatment_choice == 2]) if np.any(treatment_choice == 2) else 0
    avg_R0_combined = np.mean(R0_values[treatment_choice != 0]) if np.any(treatment_choice != 0) else 0
    
    total_R = np.sum(np.where(treatment_choice == 0, R0_values,
                              np.where(treatment_choice == 1, R1_values, R2_values)))
    total_C = np.sum(np.where(treatment_choice == 0, C0_base,
                              np.where(treatment_choice == 1, C1_T1, C1_T2)))
    
    return treatment_choice#, avg_R0_T1, avg_R0_T2, avg_R0_combined, total_R, total_C


######################

def Plotting(learner_results, summary_df, decisions):
    R0_values = learner_results['Tx_1_vs_Control']['control_pred']
    R1_values = learner_results['Tx_1_vs_Control']['treatment_pred']
    R2_values = learner_results['Tx_2_vs_Control']['treatment_pred']
    
    # # Plot Treatment effectiveness
    # plt.figure(figsize=(12, 8))
    # plt.plot(R0_values, R1_values, label='T1 Effectiveness')
    # plt.plot(R0_values, R2_values, label='T2 Effectiveness')
    # plt.plot(R0_values, R0_values, label='R0', linestyle='--')
    # plt.xlabel('Baseline Risk (R0)')
    # plt.ylabel('Risk After Treatment')
    # plt.title('Effectiveness of T1 and T2 Treatments')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    # Plot Number Treated vs k
    plt.figure(figsize=(12, 6))
    plt.plot(summary_df['k'], summary_df['Num Tx T1'], marker='o', label='T1')
    plt.plot(summary_df['k'], summary_df['Num Tx T2'], marker='o', label='T2')
    plt.plot(summary_df['k'], summary_df['Num Tx Total'], marker='o', label='Total Treated')
    plt.xlabel('k')
    plt.ylabel('Number Treated')
    plt.title('Number Treated vs k')
    plt.legend()
    plt.grid(True)
    plt.show()

    #Plot R0 Across Treatment
    plt.figure(figsize=(12, 6))
    plt.plot(summary_df['k'], summary_df['Avg R0 T1'], marker='o', label='T1')
    plt.plot(summary_df['k'], summary_df['Avg R0 T2'], marker='o', label='T2')
    plt.plot(summary_df['k'], summary_df['Avg R0 No Tx'], marker='o', label='Not Treated')
    plt.xlabel('k')
    plt.ylabel('Average Risk (Base)')
    plt.title('Average Risk (Base) vs k')
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

    # Plot Treatment Decision Heatmap
    # plt.figure(figsize=(12, 8))
    # plt.imshow(decisions, extent=[min(R0_values), max(R0_values), 0, 1], origin='lower', aspect='auto', cmap='viridis')
    # plt.colorbar(ticks=[0, 1, 2], label='Treatment Decision')
    # plt.clim(-0.5, 2.5)
    # plt.xlabel('Baseline Risk (R0)')
    # plt.ylabel('k (outcome vs. cost weight)')
    # plt.title('Optimal Treatment Decision based on Baseline Risk and k')
    # plt.show()

def plot_cost_comparison(R0_values, C0_adjusted, C1_T1, C1_T2):
    # Ensure all arrays are 1D
    R0_values = np.array(R0_values).flatten()
    C0_adjusted = np.array(C0_adjusted).flatten()
    C1_T1 = np.array(C1_T1).flatten()
    C1_T2 = np.array(C1_T2).flatten()
    
    # Ensure all arrays have the same length
    if not (R0_values.shape == C0_adjusted.shape == C1_T1.shape == C1_T2.shape):
        raise ValueError("R0_values, C0_adjusted, C1_T1, and C1_T2 must all have the same length")
    
    # Plot C0_adjusted, C1_T1, and C1_T2
    plt.figure(figsize=(10, 6))

    # Plot each cost curve against the R0 (cohort) values
    plt.plot(R0_values, C0_adjusted, label="C0 Adjusted", marker='o', color='b')
    plt.plot(R0_values, C1_T1, label="C1 (T1 Cost)", marker='x', color='orange')
    plt.plot(R0_values, C1_T2, label="C2 (T2 Cost)", marker='s', color='green')

    # Add labels and title
    plt.xlabel('Cohort (R0 values or Risk Percentiles)')
    plt.ylabel('Cost')
    plt.title('C0 Adjusted vs C1 (T1) vs C2 (T2) Cost Comparison')
    plt.legend()

    # Display the grid and plot
    plt.grid(True)
    plt.show()




######################
# Main Prog

def main_prog(train_df, test_df, features, outcomeCol, treatmentCol, C0_base, epsilon, alpha, cost_factors, percentiles_C, learner_results):
    k_range = np.linspace(0, 1, 100)
    decisions = []
    
    for k_val in k_range:
        treatment_choice = evaluate_treatments(learner_results, k_val, alpha, C0_base, cost_factors, percentiles_C)
        decisions.append(treatment_choice)
    
    decisions = np.array(decisions)
    
    summary_data = []
    R0_values = learner_results['Tx_1_vs_Control']['control_pred']
    R1_values = learner_results['Tx_1_vs_Control']['treatment_pred']
    R2_values = learner_results['Tx_2_vs_Control']['treatment_pred']
 
    for k_val in np.linspace(0, 1, 11):  # 11 points for summary
        idx = np.argmin(np.abs(k_range - k_val))
        treatment_choice = decisions[idx]
        
        num_T1 = np.sum(treatment_choice == 1)
        num_T2 = np.sum(treatment_choice == 2)
        avg_R0_T1 = np.mean(R0_values[treatment_choice == 1]) if num_T1 > 0 else 0
        avg_R0_T2 = np.mean(R0_values[treatment_choice == 2]) if num_T2 > 0 else 0
        avg_R0_no_tx = np.mean(R0_values[treatment_choice == 0]) if np.sum(treatment_choice == 0) > 0 else 0
        
        # Calculate Total R
        total_R = np.sum(np.where(treatment_choice == 0, R0_values,
                                  np.where(treatment_choice == 1, R1_values, R2_values)))
        
        # Calculate Total C
        C0 = C0_adjust(R0_values, C0_base)
        C1_T1 = C1(R0_values, 'T1', percentiles_C, cost_factors)
        C1_T2 = C1(R0_values, 'T2', percentiles_C, cost_factors)
        total_C = np.sum(np.where(treatment_choice == 0, C0_base,
                                  np.where(treatment_choice == 1, C1_T1, C1_T2)))
        
        summary_data.append({
            'k': k_val,
            'Num Tx T1': num_T1,
            'Num Tx T2': num_T2,
            'Num Tx Total': num_T1 + num_T2,
            'Avg R0 No Tx': avg_R0_no_tx,
            'Avg R0 T1': avg_R0_T1,
            'Avg R0 T2': avg_R0_T2,
            'Total R': total_R,
            'Total C': total_C
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Plotting
    Plotting(learner_results, summary_df, decisions)
    plot_cost_comparison(R0_values, C0, C1_T1, C1_T2)
    
    print("Summary of Optimal Treatment Choices:")
    print(summary_df.to_string(index=False, float_format='{:,.2f}'.format))

    #return decisions, summary_df