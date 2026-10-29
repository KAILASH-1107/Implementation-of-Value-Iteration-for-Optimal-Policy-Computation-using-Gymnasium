# Implementation-of-Value-Iteration-for-Optimal-Policy-Computation-using-Gymnasium

---
## Aim

To implement the **Value Iteration** algorithm for solving a finite Markov Decision Process using the Gymnasium `FrozenLake-v1` environment, and to compute the optimal state-value function and optimal policy using the Bellman optimality equation.

---

## Problem Statement
Implement the Value Iteration algorithm on the FrozenLake environment to determine the optimal value of each state and derive the optimal policy that maximizes the expected cumulative reward. The algorithm should repeatedly update state values until convergence and then extract the best action for every state.




## Software Requirements
```
Python 
Gymnasium Library
NumPy
Jupyter Notebook/Google colab
```
## Environment Description

The environment is a 4×4 customized FrozenLake consisting of safe frozen cells (F), hole states (H), a start state (S), and a goal state (G). The agent begins at the Start (S) and must reach the Goal (G) while avoiding the holes. Since the environment uses is_slippery=True, the agent may not always move in the intended direction, making the environment stochastic. The Value Iteration algorithm is used to compute the optimal state-value function and the optimal policy for safely reaching the goal.
### Map Used:
```

F F H F
H S H F
H F F H
H F G H

```
```
Where:

S – Start State
F – Frozen Surface (Safe)
H – Hole (Terminal State)
G – Goal State (Reward = 1)

Since is_slippery=True, the agent may move in unintended directions due to slippery ice, making the environment stochastic. 
```
## MDP Representation

#### States (S)
There are 16 states in a 4×4 FrozenLake grid.

S={0,1,2,…,15}
Each grid cell represents one state. The agent starts from the Start (S) state and aims to reach the Goal (G) state while avoiding Hole (H) states.

#### Actions (A)

There are four possible actions.
```
Action	Value
Left	0
Down	1
Right	2
Up	3
```
### Transition Probability (P)
The transition probability defines the likelihood of moving from one state to another after taking an action.
Since is_slippery=True, the environment is stochastic, meaning the agent may move in an unintended direction. Therefore, the next state depends on both the selected action and the slipping probability.
``` 
P(s′∣s,a)
```

### Reward Function (R)
```
Goal State = +1
Hole = 0
Frozen State = 0
Discount Factor (γ)
γ=0.99
```
The discount factor gives high importance to future rewards.

## Theory
Value Iteration is a dynamic programming algorithm used to solve Markov Decision Processes (MDPs) by repeatedly applying the Bellman Optimality Equation until the value function converges. In each iteration, the algorithm updates the value of every state by selecting the action that provides the maximum expected future reward. Once the values converge, the optimal policy is obtained by choosing the action with the highest value for each state.
The Bellman Optimality Equation is:

```
V(s) = max_a Σ P(s' | s, a) [R(s, a, s') + γV(s')]
```

## Algorithm

1. Create the customized FrozenLake-v1 environment using the modified map.
2. Initialize the value of all states to zero.
3. Set the discount factor (γ = 0.99) and convergence threshold (θ = 1e−8).
4. Compute the expected value for every possible action in each state using the Bellman Optimality Equation.
5. Update each state's value with the maximum expected action value.
6. Repeat the updates until the change in state values is less than the convergence threshold.
7. Extract the optimal policy by selecting the action with the highest expected value for each state.
8. Display the number of iterations, optimal state-value function, and optimal policy.



## Python Program

```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
# -------------------------------------------------
# Create FrozenLake Environment
# -------------------------------------------------
env_desc = [
    "FFHF",
    "HSHF",
    "HFFH",
    "HFGH"
]

env = gym.make("FrozenLake-v1", desc=env_desc, is_slippery=True)
# -------------------------------------------------
# Value Iteration Algorithm
# -------------------------------------------------


def value_iteration(env, gamma=0.99, theta=1e-8):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    P = env.unwrapped.P
    
    V = np.zeros(n_states)
    iteration = 0

    while True:

        delta = 0
        policy = np.zeros(n_states, dtype=int)
        for s in range(n_states):
            v = V[s]
            action_values = []

            for a in range(n_actions):
                value = 0

                for prob, next_state, reward, done in P[s][a]:
                    value += prob * (reward + gamma * V[next_state] * (not done))
                    
                action_values.append(value)
            policy[s] = np.argmax(action_values)
            
            V[s] = max(action_values)
            delta = max(delta, abs(v - V[s]))
            

        iteration += 1
        
        if delta < theta:
            break
    return V, policy, iteration
# -------------------------------------------------  
# Run Value Iteration
# -------------------------------------------------


V, policy, iterations = value_iteration(env)

# -------------------------------------------------
# Display Output
# -------------------------------------------------
print("Name: Kailash V")
print("Register Number: 212224240067")
print("Value Iteration Completed")
print("Number of Iterations:", iterations)

print("\nOptimal State-Value Function:")
print(np.round(V.reshape(4, 4), 4))


action_symbols = {
    0: "L",
    1: "D",
    2: "R",
    3: "U"
}

policy_grid = np.array(
    [action_symbols[action] for action in policy]
).reshape(4, 4)

print("\nOptimal Policy:")
print(policy_grid)

env.close()
```

---

## Output


<img width="760" height="290" alt="image" src="https://github.com/user-attachments/assets/f5b53440-529b-47d1-8fb1-01aa3aa5de1d" />


<img width="606" height="360" alt="image" src="https://github.com/user-attachments/assets/9a02c421-0c26-4abd-ae4b-2eb8a1cc7534" />






## Result

The Value Iteration algorithm was successfully implemented on the customized FrozenLake-v1 environment. It converged after several iterations and computed the optimal state-value function and optimal policy for the modified map. The resulting policy guides the agent from the start state to the goal while avoiding hole states as much as possible.

## Inference
The comparison between the original and modified FrozenLake environments shows that changing the map layout directly affects the Markov Decision Process. In the modified environment, the start state, goal state, and hole positions were rearranged, creating different state transitions and fewer safe paths. As a result, the optimal state-value function, policy, and convergence behavior differed from the original environment. The Value Iteration algorithm successfully adapted to these changes by recalculating the optimal values and selecting the best action for each state. This confirms that Value Iteration is robust and capable of finding the optimal policy for different MDP configurations.
