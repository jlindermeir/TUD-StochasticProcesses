import pathlib

import numpy as np
import matplotlib.pyplot as plt

# Define parameters
N_STEPS = 100
N_REALIZATIONS = 100
STEP_LENGTH = 1
P_LIST = [0.2, 0.5, 0.9]
COLOR_LIST = ['r', 'g', 'b']


def random_walk(p_right, step_length=STEP_LENGTH, n_steps=N_STEPS):
    # Draw random variables from {-1, 1}
    random_arr = np.random.choice([-1, 1], size=n_steps, p=[1 - p_right, p_right])
    # Add up the individual steps
    return np.cumsum(step_length * np.concatenate([[0], random_arr]))


# Set random seed
np.random.seed(69)

# Create a plot
fig, (ax, var_ax) = plt.subplots(2, 1, tight_layout=True, figsize=(6, 8), sharex='all')

ax.grid()
ax.set_ylabel("Position $x_i$")

var_ax.grid()
var_ax.set_xlabel("Step $i$")
var_ax.set_ylabel("Variance $Var(x_i)$")

# Plot walks for each probability
for p_right, color in zip(P_LIST, COLOR_LIST):
    walk_list = np.array([
        random_walk(p_right)
        for _ in range(N_REALIZATIONS)
    ])

    ax.plot(walk_list.T, '-', c=color, alpha=0.05)
    ax.plot(np.mean(walk_list, axis=0), c=color,  label=f'$p = {p_right}$')
    ax.plot((2 * p_right - 1) * STEP_LENGTH * np.arange(N_STEPS + 1), c=color, ls='--')

    var_ax.plot(np.std(walk_list, axis=0) ** 2, c=color, label=f'p = {p_right}')
    var_ax.plot(4 * STEP_LENGTH * np.arange(N_STEPS + 1) * p_right * (1 - p_right), c=color, ls='--')

# Add legend and save figure
fig.legend(*ax.get_legend_handles_labels(), ncol=len(P_LIST), loc='upper center')
fig.savefig(pathlib.Path(__file__).parent / 'random_walk.pdf')
plt.show()
