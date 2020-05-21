## chapter1 Introduction

#### 1.1 lead in RL:
+ **target**: how to map situations($s$) to actions($a$), so as to maximize a numerical reward signal.
+ **method**: discover which actions yield the most reward by *trying them*.
+ **challenges**: actions may affect next situation and throught that, all subsequent rewards.
+ **characteristics**: *trial-and-error search* and *delayed reward*.
+ **basic elements**:
  - sensation: agent must be able to sense the state of the environment to some extent
  - action: agent must be able to take actions that *affect* the state
  - goal: agent must have goals related to the state of environment.
$\bigstar$a suitable RL method should at least include these three aspects.

understanding supervised learning under RL framework
+ samples ($X$, $y$): $X$ is description of a situation, $y$ is a specification of a correct action the system should take to that situation.
+ whiy it's inadequate for learning from interaction?——impractical to obtain examples of all situations in which agent has to act.

#### 1.2 uniqueness of RL:
+ $exploreation$ vs. $exploitation$
+ complete, interactive, goal-seeking agent

#### 1.3 elements of RL
4 elements:
+ policy($\pi$): mapping from perceived states to actions to be taken. In general, probabilities for each action.
+ reward signal($r$): the number which sent by the environment to agent in each time step(*immediate* reward). Recall that the goal is to maximize total rewards in a long time. In general $r$ is function of state $s$ and following action $a$.
+ value function($v(s)$): total amount expected to accumulate over the future starting from state $s$. $v(s)$ indicates a *long-term* desirability of a state after accounting for the states that are likely to follow and rewards available in those states. A state which has a low $r(s)$(immediate reward) could still has a high $v(s)$ if it usually followed by high rewards states.
<font color=red>$\bigstar$ agent takes action based on $v$ instead of $r$. Although $r$ is relatively direct and $v$ must be estimated and re-estimated from sequences of observations</font>. *value estimation has a central role*.
+ model of the environment(optional): if an environment model is build, then this is model-based method and planning is used. Opposite to it, pure model-free trial-and-error method. A model might predict  the resultant next state and reward given current state and action taken.

#### 1.4 limitations and scopes
+ limitation of this book:
  - focus on $v(s)$ and $\pi(s)$, state signal making is not the concern. You can think state as whatever information is available to the agent about its environment.
  - sometimes $v(s)$ could not be the center or even unnecessary if following conditions are met:
    * space of policy is very small
    * good policies are common and easy to find
    * a lot of time is available for the search
    * agent cannot sense the complete state of its environment

#### 1.5 example: tic-tac-toe
lessons from this game:
- this is a model-free method, no model is built to construct the behavior of the opponent
- most of time greedily choosing the action that leads to the greatest state values(expolitation)
- occasionally, random selecting other states(expolration)
- update state value after each greedy move(not in random exploration), by modifying current value of the earlier state to be closer to the value of the later state, with step-size $\alpha$: $V(S_t)=V(S_t)+\alpha[V(S_{t+1})-V(S_t)]$(TD: temporal difference)
  + if $\alpha$ progressively decays to zero, then the model converges to the optimal policy with repect to this opponent
  + if $\alpha$ is not reduced all the way to zero, the agent can keep learning, in this way can also play well against opponents that slowly change their way of playing.

  instead of giving credit to all behaviors in the game after each episode, RL can evaluate individual state during the game.
- Generally, RL is applicable when:
  + behavior continuously indefinitely when rewards of various magnititudes can be received at any time.
  + very large even infinite states: in such condition, no way to explore all possible states $\rightarrow$ new states selected based on information saved from similar past experience: ability to generalize from past experience is the key, where supervised learning can be combined with RL.
  + prior information can be incorporated with RL.
  + model-free methods can be important block of model-based methods.

  #### 1.6 brief history about RL
  ```mermaid
  graph LR
  A[trial-and-error] -->D(1980s)
  B[optimal control: solved by value function & dynamic programming <br/>&emsp;+ Bellman equation: using dynamical system's state and value function<br/>&emsp;+discrete stochastic version: Markov Decision Process a.k.a MDP<br/>*Dynamic Programming is usually incremental and iterative, <br/>gradually reach the correct answer through successive approximations]-->D
  C[temporal-difference a.k.a.TD]-->D
  D-->E(modern RL)
  ```
