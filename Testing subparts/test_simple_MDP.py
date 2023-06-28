
# simple MDP to test

#%%
import numpy as np
import matplotlib.pyplot as plt

# Define the MDP framework
num_states = 2
num_actions = 2

# Define the transition probability matrix
transition_probs = np.array([[0.8, 0.2],  # Transition probabilities from state 0 to state 0 and state 1
                            [0.3, 0.7]]) # Transition probabilities from state 1 to state 0 and state 1

# Define the reward matrix
rewards = np.array([[1, 0],  # Rewards for staying in state 0 and transitioning to state 1
                    [0, 2]]) # Rewards for staying in state 1 and transitioning to state 0

# Define the policy (action selection)
policy = np.array([[0, 1],  # Policy for state 0: stay (action 0), transition to state 1 (action 1)
                   [1, 0]]) # Policy for state 1: transition to state 0 (action 0), stay (action 1)

# Function to plot the transition probability matrix
def plot_transition_probs():
    plt.imshow(transition_probs, cmap='Blues')
    plt.colorbar()
    plt.title('Transition Probability Matrix')
    plt.xlabel('Next State')
    plt.ylabel('Current State')
    plt.xticks(np.arange(num_states))
    plt.yticks(np.arange(num_states))
    plt.show()

# Function to plot the reward matrix
def plot_rewards():
    plt.imshow(rewards, cmap='Reds')
    plt.colorbar()
    plt.title('Reward Matrix')
    plt.xlabel('Action')
    plt.ylabel('State')
    plt.xticks(np.arange(num_actions))
    plt.yticks(np.arange(num_states))
    plt.show()

# Function to plot the action matrix
def plot_actions():
    actions = np.argmax(policy, axis=1)
    
    fig, ax = plt.subplots()
    ax.imshow(actions.reshape(num_states, 1), cmap='Greens')
    ax.set_title('Action Matrix')
    ax.set_xlabel('Action')
    ax.set_ylabel('State')
    ax.set_xticks([0])
    ax.set_yticks(np.arange(num_states))
    ax.set_yticklabels(['State 0', 'State 1'])
    ax.tick_params(axis='x', length=0)
    ax.grid(color='black', lw=0.5, linestyle='-')
    
    for i in range(num_states):
        for j in range(num_actions):
            ax.text(0, i, f'{j}', ha='center', va='center', color='black')
    
    plt.show()

# Function to perform an action and observe the next state and reward
def take_action(state, action):
    next_state = np.random.choice(num_states, p=transition_probs[state][action])
    reward = rewards[state][action]
    return next_state, reward

# Function to simulate the MDP framework
def simulate_mdp(num_steps):
    state = np.random.choice(num_states)  # Choose a random initial state
    
    for step in range(num_steps):
        print(f"Step {step + 1} - Current State: {state}")
        
        action = policy[state]
        next_state, reward = take_action(state, action)
        
        print(f"Action: {action}, Next State: {next_state}, Reward: {reward}")
        
        state = next_state

# Plot the transition probability matrix
plot_transition_probs()

# Plot the reward matrix
plot_rewards()

# Plot the action matrix
plot_actions()

# Simulate the MDP for a given number of steps
num_steps = 5
simulate_mdp(num_steps)

#%%