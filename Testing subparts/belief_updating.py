# belief updating ideas

import numpy as np

# Step 1: Define the initial belief
belief = np.array([0.5, 0.5])  # Equal initial probabilities for each state

# Step 2: Observe evidence
observation = 'A'  # Example observation, can be any value

# Step 3: Update the belief using Bayes' rule
transition_matrix = np.array([[0.7, 0.3],  # Transition probabilities from state A to A and B
                             [0.2, 0.8]])  # Transition probabilities from state B to A and B

observation_model = {'A': np.array([0.6, 0.4]),  # Observation model for state A
                     'B': np.array([0.3, 0.7])}  # Observation model for state B

# Compute the likelihood
likelihood = observation_model[observation]

# Compute the prior
prior = belief

# Compute the unnormalized posterior
unnormalized_posterior = likelihood * prior

# Normalize the posterior
posterior = unnormalized_posterior / np.sum(unnormalized_posterior)

# Step 4: Repeat the process with new observations
# You can update the belief by iterating the above steps with new observations
# For example, let's assume a new observation is received
new_observation = 'B'

# Compute the likelihood for the new observation
new_likelihood = observation_model[new_observation]

# Update the prior with the previous posterior
prior = posterior

# Compute the unnormalized posterior for the new observation
new_unnormalized_posterior = new_likelihood * prior

# Normalize the posterior
new_posterior = new_unnormalized_posterior / np.sum(new_unnormalized_posterior)

# The new_posterior represents the updated belief after incorporating the new observation
