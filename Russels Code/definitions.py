#%%

# incorporating the reward function made in reward_function.ipynb

import matplotlib.pyplot as plt
import math
import ipywidgets as widgets
from IPython.display import display, clear_output


import matplotlib.pyplot as plt
import math
import numpy as np


def rewards_function(steps):
    # Function Definitions
    def rewardUpdate(x0, tau, n, prev):
        if n == 0:
            new_r = prev
        elif n == 1:
            new_r = x0
        else:
            new_r = x0 * math.exp(-tau * n)
        return new_r
    def consecutiveCounter(seq, val, idx):
        count = 0
        if seq[idx] != val:
            pass
        else:
            for state in reversed(seq[: idx + 1]):
                if state == val:
                    count += 1
                else:
                    break
        return count
    # Initial Settings
    initial_reward = 0.75  # should change it back to 1
    r0 = [initial_reward]
    r1 = [initial_reward]
    tau = 0.5
    s = []
    for i in range(steps):
        s.append(np.random.choice([0, 1]))
        print(s)
        # New Settings
        r0 = [initial_reward]
        r1 = [initial_reward]
        r0_counter = 0
        r1_counter = 0
        for i, state in enumerate(s):
            r0_counter = consecutiveCounter(s, 0, i)
            r1_counter = consecutiveCounter(s, 1, i)
            new_r0 = rewardUpdate(1, tau, r0_counter, r0[-1])
            r0.append(new_r0)
            new_r1 = rewardUpdate(1, tau, r1_counter, r1[-1])
            r1.append(new_r1)
        # Plot
        plt.scatter(range(len(r0)), r0, color="r", alpha=0.5)
        plt.scatter(range(len(r1)), r1, color="b", alpha=0.3)
        plt.show()
rewards_function(6)
# %%
