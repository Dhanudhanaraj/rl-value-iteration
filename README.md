# VALUE ITERATION ALGORITHM

## AIM
To develop a Python program to find the optimal policy for the given MDP using the value iteration algorithm.

## PROBLEM STATEMENT
The FrozenLake environment in OpenAI Gym is a gridworld problem that challenges reinforcement learning agents to navigate a slippery terrain to reach a goal state while avoiding hazards. Note that the environment is closed with a fence, so the agent cannot leave the gridworld.

## VALUE ITERATION ALGORITHM
Value iteration is a method of computing an optimal MDP policy and its value.
It begins with an initial guess for the value function, and iteratively updates it towards the optimal value function, according to the Bellman optimality equation.
The algorithm is guaranteed to converge to the optimal value function, and in the process of doing so, also converges to the optimal policy.
The algorithm is as follows:

1.Initialize the value function V(s) arbitrarily for all states s.

2.Repeat until convergence:

Initialize aaction-value function Q(s, a) arbitrarily for all states s and actions a.
For all the states s and all the action a of every state:
Update the action-value function Q(s, a) using the Bellman equation.
Take the value function V(s) to be the maximum of Q(s, a) over all actions a.
Check if the maximum difference between Old V and new V is less than theta.
Where theta is a small positive number that determines the accuracy of estimation. 3.If the maximum difference between Old V and new V is greater than theta, then
Update the value function V with the maximum action-value from Q.
Go to step 2. 4.The optimal policy can be constructed by taking the argmax of the action-value function Q(s, a) over all actions a. 5.Return the optimal policy and the optimal value function.

This MDP algorithmn is written for Goal state of (3,0) with state number as 12.

## VALUE ITERATION FUNCTION
```
NAME:DHANUMALYA.D
REGISTER NUMBER:212222230030
```
```
## Code for Goal state as (3,0) with state number as 12
desc=['SFHF','FFFH','FFHF','GFFH']
env=gym.make('FrozenLake-v1',desc=desc)
init_state=env.reset()
goal_state=12
P=env.env.P
init_state
```
```
## Value Iteration Algorithm
def value_iteration(P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    while True:
      Q=np.zeros((len(P),len(P[0])),dtype=np.float64)
      for s in range(len(P)):
        for a in range(len(P[s])):
          for prob,next_state,reward,done in P[s][a]:
            Q[s][a]+=prob*(reward+gamma*V[next_state]*(not done))
      if np.max(np.abs(V-np.max(Q,axis=1)))<theta:
        break
      V=np.max(Q,axis=1)
    pi=lambda s:{s:a for s,a in enumerate(np.argmax(Q,axis=1))}[s]
    return V, pi
```


## OUTPUT:
## Optimal policy
![Screenshot 2024-03-13 182918](https://github.com/Dhanudhanaraj/rl-value-iteration/assets/119218812/a4ac2d6a-162b-4aa4-88db-0239cd6a3bc1)

## Optimal value function
![Screenshot 2024-03-13 182936](https://github.com/Dhanudhanaraj/rl-value-iteration/assets/119218812/7a9aa36d-1135-438c-9e32-202108f638bf)

## Success rate for the optimal policy.
![Screenshot 2024-03-13 182928](https://github.com/Dhanudhanaraj/rl-value-iteration/assets/119218812/e8abfca1-6e3c-4ce8-a958-48a9c67187e9)

## RESULT:
Thus, a Python program is developed to find the optimal policy for the given MDP using the value iteration algorithm.
