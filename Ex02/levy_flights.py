import numpy as np
import matplotlib.pyplot as plt


# Probability distributions of phi and Y
def sample_phi_y(size):
    return np.array([
        np.random.uniform(low=-np.pi / 2, high=np.pi / 2, size=size),
        np.random.exponential(1, size=size)
    ])


# Define random variable X
def X(phi, Y, alpha):
    return (
        np.sin(alpha * phi)
        / np.cos(phi) ** (1 / alpha)
        * (
            np.cos((1 - alpha) * phi)
            / Y
        ) ** ((1 - alpha) / alpha)
    )


if __name__ == '__main__':
    for alpha in np.linspace(1, 2, 5):
        # Random walk
        n_steps = 5000
        step_arr = X(*sample_phi_y((n_steps, 2)), alpha)
        pos_arr = np.cumsum(step_arr, axis=0)

        fig, ax = plt.subplots(tight_layout=True)
        ax.grid()
        ax.axis('equal')
        ax.set_title(f'$N = {n_steps}$, $\\alpha = {alpha}$')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.plot(*pos_arr.T, 'k-')

    # PDF
    alpha = 1.7
    n_trials = 5000
    sample_X_arr = X(*sample_phi_y(n_trials), alpha)
    plt.figure()
    plt.hist(sample_X_arr, bins=100)
    plt.show()

