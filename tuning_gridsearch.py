import numpy as np
import pandas as pd

# Fixed parameters
hop_max = 2
p_i = 0.045
e_res = 0.5
num_neighbors = 5
sigma_e = 0.1

# Utility function
def calculate_utility(alpha, beta, m, p_i, e_res, num_neighbors, sigma_e):
    ctb_benefit = beta * num_neighbors
    e_balance_benefit = alpha / sigma_e if sigma_e != 0 else 0
    e_cost = p_i / (m * e_res) if m * e_res != 0 else float('inf')
    return ctb_benefit - e_balance_benefit - e_cost

# Parameter ranges
alpha_range = np.arange(0.5, 2.6, 0.01)
beta_range = np.arange(0.5, 2.6, 0.01)
m_range = np.arange(0.005, 0.1, 0.001)

# Grid search over ALPHA, BETA, and M
results = []
best_utility = float('-inf')
best_params = None

for alpha in alpha_range:
    for beta in beta_range:
        for m in m_range:
            utility = calculate_utility(alpha, beta, m, p_i, e_res, num_neighbors, sigma_e)
            results.append({'ALPHA': alpha, 'BETA': beta, 'M': m, 'Utility': utility})
            if utility > best_utility:
                best_utility = utility
                best_params = {'ALPHA': alpha, 'BETA': beta, 'M': m}

# Save results to CSV
df = pd.DataFrame(results)
df.to_csv('utility_grid_all_params.csv', index=False)

# Print the best parameters and utility
print(f"Best parameters: ALPHA={best_params['ALPHA']}, BETA={best_params['BETA']}, M={best_params['M']}")
print(f"Highest utility: {best_utility:.4f}")