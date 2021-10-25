
import pathlib
import numpy as np
import pylab
import random
import matplotlib.pyplot as plt

# defining the number of steps, L and b
N_STEPS = 100
N_REALIZATIONS = 100
L_LIST = [1, 1, 3]
b_LIST = [0.5, 0.1, 0.1]
COLOR_LIST = ['r', 'g', 'b']

def random_walk(b, L, n_steps=N_STEPS):
    # Draw random variables from {-1, 1}
    random_arr = b*np.random.random_sample((N_STEPS,))+L
    # Add up the individual steps
    return np.cumsum(np.concatenate([[0], random_arr]))


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
for b, L, color in zip(b_LIST, L_LIST, COLOR_LIST):
    walk_list = np.array([
        random_walk(b, L)
        for _ in range(N_REALIZATIONS)
    ])

    ax.plot(walk_list.T, '-', c=color, alpha=0.05)
    ax.plot(np.mean(walk_list, axis=0), c=color,  label=f'$L = {L}$'+f', $b = {b}$')
    ax.plot(L* np.arange(N_STEPS + 1), c=color, ls='--')

    var_ax.plot(np.std(walk_list, axis=0) ** 2, c=color, label=f'$L = {L}$'+f', $b = {b}$')
    var_ax.plot(np.arange(N_STEPS + 1)**2 * b**2, c=color, ls='--')

# Add legend and save figure
fig.legend(*ax.get_legend_handles_labels(), ncol=len(b_LIST), loc='upper center')
fig.savefig(pathlib.Path(__file__).parent / 'random_walkII.pdf')
plt.show()
