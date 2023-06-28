# just gathering ideas to create POMDP code from scratch


import math
import numpy as np
import matplotlib
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sb

class POMDP(): #base class

    #define main attributes/properties

    def __init__(self):
        self.actions = ['l_att','right_att'] #would probably be better to assign those strings to numbers, maybe better dictionary?
        self.states = ['left','right']       
        self.transition_probability = 0.8 #changeable
        # self.b0_left = 0
        # self.b0_right = 0
        # self.switching_cost = -5

    # define methods (function)

    def rewards(self,t, t_ls, tau_r):
        # depends on decaying function 
        return math.exp(-(t-t_ls)/tau_r)   #t_ls is time of last time step and tau_r is the time constant of exponential decay
        #add visualizer

    def update_belief():
        mu = 1
        sigma = 0.6
        sigma_att = 0
        observation_value = np.random.normal(mu, sigma, 1) # say left eye get 1 (mu = 1), and right eye -1 (mu = -1)
        belief_prob = norm.pdf(observation_value, loc=mu, scale=sigma_att)
        return 
        #add visualizer

    def value_function():
        return 
            #add visualizer

    def decision_maker():
        #compare value function
        pass
            #add visualizer





# test belief update shervin

#%%


mu = 1
sigma = 0.6
sigma_att = 0
observation_value = np.random.normal(mu, sigma, 1) # say left eye get 1 (mu = 1), and right eye -1 (mu = -1)
belief_prob = norm.pdf(observation_value, loc=mu, scale=sigma_att)

#%%


# define solver for Value Iteration:


# Value Iteration Visualization

# plot_pomdp_utility(utility)




### VISUALIZATION

# transition matrix


# reward graph

# Data visualization: belief, reward and action 
# → best way to represent transition probability distribution and reward probability distribution is with big three-dimensional tables 
