import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Fixed parameters
hop_max = 2
p_i = 0.045
e_res = 0.5
num_neighbors = 10
sigma_e = 0.1
default_alpha = 1.5
default_beta = 1.5
default_m = 0.01

# Utility function
def calculate_utility(alpha, beta, m, p_i, e_res, num_neighbors, sigma_e):
    ctb_benefit = beta * num_neighbors
    e_balance_benefit = alpha / sigma_e if sigma_e != 0 else 0
    e_cost = p_i / (m * e_res) if m * e_res != 0 else float('inf')
    return ctb_benefit - e_balance_benefit - e_cost

# Parameter ranges
alpha_range = np.arange(0.5, 2.6, 0.01)
beta_range = np.arange(0.5, 2.6, 0.01)
m_range = np.arange(0.005, 0.051, 0.001)

# Store results
results = []

# Vary ALPHA, fix BETA and M
alpha_utilities = []
for alpha in alpha_range:
    utility = calculate_utility(alpha, default_beta, default_m, p_i, e_res, num_neighbors, sigma_e)
    alpha_utilities.append(utility)
    results.append({'Parameter': 'ALPHA', 'Value': alpha, 'Utility': utility})

# Vary BETA, fix ALPHA and M
beta_utilities = []
for beta in beta_range:
    utility = calculate_utility(default_alpha, beta, default_m, p_i, e_res, num_neighbors, sigma_e)
    beta_utilities.append(utility)
    results.append({'Parameter': 'BETA', 'Value': beta, 'Utility': utility})

# Vary M, fix ALPHA and BETA
m_utilities = []
for m in m_range:
    utility = calculate_utility(default_alpha, default_beta, m, p_i, e_res, num_neighbors, sigma_e)
    m_utilities.append(utility)
    results.append({'Parameter': 'M', 'Value': m, 'Utility': utility})

# Save results to CSV
df = pd.DataFrame(results)
df.to_csv('utility_variation.csv', index=False)

# Calculate sensitivity
def calculate_sensitivity(values, utilities):
    if len(values) > 1:
        delta_u = np.diff(utilities) / np.diff(values)
        avg_sensitivity = np.mean(np.abs(delta_u))
        return avg_sensitivity
    return 0

sensitivity_alpha = calculate_sensitivity(alpha_range, alpha_utilities)
sensitivity_beta = calculate_sensitivity(beta_range, beta_utilities)
sensitivity_m = calculate_sensitivity(m_range, m_utilities)

print(f"Sensitivity of utility to ALPHA: {sensitivity_alpha:.4f} per unit")
print(f"Sensitivity of utility to BETA: {sensitivity_beta:.4f} per unit")
print(f"Sensitivity of utility to M: {sensitivity_m:.4f} per unit")

# Plot 1: Utility vs ALPHA
plt.figure(figsize=(8, 6))
plt.plot(alpha_range, alpha_utilities, marker='o', color='blue', label='Utility')
plt.xlabel('ALPHA')
plt.ylabel('Utility')
plt.title('Utility vs ALPHA (BETA=1.5, M=0.01)')
plt.grid(True)
plt.legend()
plt.savefig('utility_alpha.png')
plt.close()

# Plot 2: Utility vs BETA
plt.figure(figsize=(8, 6))
plt.plot(beta_range, beta_utilities, marker='o', color='green', label='Utility')
plt.xlabel('BETA')
plt.ylabel('Utility')
plt.title('Utility vs BETA (ALPHA=1.5, M=0.01)')
plt.grid(True)
plt.legend()
plt.savefig('utility_beta.png')
plt.close()

# Plot 3: Utility vs M
plt.figure(figsize=(8, 6))
plt.plot(m_range, m_utilities, marker='o', color='red', label='Utility')
plt.xlabel('M')
plt.ylabel('Utility')
plt.title('Utility vs M (ALPHA=1.5, BETA=1.5)')
plt.grid(True)
plt.legend()
plt.savefig('utility_m.png')
plt.close()