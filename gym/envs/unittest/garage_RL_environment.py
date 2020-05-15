# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:45:12 2020

@author: cesa_
"""
import gym
from gym import spaces
from garage_demand import demand_static, demand_stochastic
from garage_cost import Exp_cost, opex
#from garage_ENPV_obj_arrayinput import cc_start
import pandas as pd
import numpy as np
from gym.spaces import Discrete, Box
from gym.utils import seeding

# Parameters
T=20 #years
cc = 16000# Construction cost per parking space
cl = 3600000# Annual leasing land cost
#p = 0.00# Percentage of construction cost to acquire flexibility (i.e. fatter columns). Note that this is set to 0 to determine how much the flexibility is worth. The real value can be found by subracting the acquiistion cost from the NPV for a particular design.
cr = 2000# Operating cost per parking space
#ct = []# Total construction cost
gc = 0.10# Growth in construction cost per floor above two floors
n0 = 200# Initial number of parking space per floor
p = 10000# Price per parking space
r = 0.12# Discount rate
fmin = 2# Minimum number of floors built
fmax = 9# Maximum number of floors built

years = ['0' ,'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16','17','18','19','20']
k = pd.Series(index=years)
demand_projections = pd.Series(index=years)
for i in range(0,T+1):
     demand_projections[i] = demand_static(i)
     
cc_initial = 2*n0*cc

     
#Actions
NOTHING = 0
EXPAND = 1

class Garage_Env(gym.Env):

    def __init__(self):
        # Demand and capacity data
        max_capacity = 1800
        min_capacity = 200
        max_demand = np.max(demand_projections)
        min_demand = np.min(demand_projections)
        self.low = np.array([min_capacity, min_demand], dtype=np.float32)
        self.high = np.array([max_capacity, max_demand], dtype=np.float32)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)
        self.action_space = spaces.Discrete(2) # Either increase by 200 or mantain for now
        self.count = 0
        self.time_steps = 0
        self.fixed_cost = 3600000
        self.current_capacity = 200
        self.current_demand = demand_static(0)
        self.state = self.current_capacity, self.current_demand

    
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        
        current_capacity, current_demand = self.state
        if action == 0:
            possible_next_capacity = current_capacity
            E_cost = 0 
        elif action == 1:
           possible_next_capacity = current_capacity + 200
           if possible_next_capacity < 1800:         
              E_cost = Exp_cost(current_capacity,1)
           else : 
              E_cost = 0
              possible_next_capacity = 1800
               
        self.time_steps +=1       
        self.current_capacity = possible_next_capacity
        opex_val = opex(self.current_capacity)
        self.current_demand = demand_static(self.time_steps) # bring forward timestep
        self.state = (self.current_capacity, self.current_demand)
        
        if self.time_steps == T:
            done = True
        else :
            done = False
        
        revenue = np.minimum(self.current_capacity, self.current_demand)*p
        reward = revenue - opex_val - self.fixed_cost -E_cost

        return np.array(self.state), reward, done, {}
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def render(self, mode='human'):
        pass
        
    def reset(self):
        self.current_capacity = 200
        self.current_demand = demand_static(0)
        self.state = (self.current_capacity, self.current_demand)
        return np.array(self.state)
    
    

    
def agent_policy(rand_generator):
    """
    Given random number generator and state, returns an action according to the agent's policy either 0 or 1.
    Returns:
        chosen action [int]
    """
    
    # set chosen_action as 0 or 1 with equal probability
    # state is unnecessary for this agent policy
    chosen_action = rand_generator.choice([0,1])
    
    return chosen_action 


 # some smoke testing
#env_test = Garage_Env()
#observation = env_test.reset()

#returns = 0
#for step in range(40):
  #action = agent_policy(np.random.RandomState())
  #observation, reward, done, _ = env_test.step(action)
  #returns += reward/(1.12**step) # using discount rate of .12
  #print("step %i: action=%i, capacity=%i , demand=%i => reward = %i, done = %s" % (step, action, observation[0], observation[1], reward, done))
  #if done: break       
    
#print("Approximate NPV", returns - cc_initial) 



#env = Garage_Env()
#env = make_vec_env(lambda: env, n_envs=1)


# Train the agent
#model = ACKTR('MlpPolicy', env, verbose=1).learn(5000)


# Test the trained agent
#obs = env.reset()
#n_steps = 20
#for step in range(n_steps):
  #action, _ = model.predict(obs, deterministic=True)
 # print("Step {}".format(step + 1))
  #print("Action: ", action)
  #obs, reward, done, info = env.step(action)
  #print('obs=', obs, 'reward=', reward, 'done=', done)
 # env.render(mode='human')
  #if done:
    # Note that the VecEnv resets automatically
    # when a done signal is encountered
    #print("Goal reached!", "reward=", reward)
    #break