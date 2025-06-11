import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read results
df = pd.read_csv('/Users/richard/parallel-computing-and-multimedia-processing/srs/coursework/results/wann_fitness.csv')

# Calculate speedup (baseline is 1 process time)
baseline_time = df[df['Process_Count'] == 1]['Execution_Time'].iloc[0]
df['Speedup'] = baseline_time / df['Execution_Time']
df['Efficiency'] = df['Speedup'] / df['Process_Count']

print("Scalability Results:")
print(df)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Speedup plot
ax1.plot(df['Process_Count'], df['Process_Count'], '--', color='gray', label='Ideal Speedup')
ax1.plot(df['Process_Count'], df['Speedup'], 'o-', label='Actual Speedup')
ax1.set_xlabel('Number of Processes')
ax1.set_ylabel('Speedup Factor')
ax1.set_title('Parallel Speedup')
ax1.legend()
ax1.grid(True)

# Efficiency plot
ax2.plot(df['Process_Count'], df['Efficiency'], 'o-', color='red')
ax2.axhline(y=1.0, color='gray', linestyle='--', label='Perfect Efficiency')
ax2.set_xlabel('Number of Processes')
ax2.set_ylabel('Efficiency')
ax2.set_title('Parallel Efficiency')
ax2.set_ylim(0, 1.2)
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('actual_scalability_results.png', dpi=300)
plt.show()
