
import random
import gym
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import time



class agent():

    def __init__(self, obs_dim = 4, act_dim = 1, control_points = {2:0.00}, train_freq = 100, model_lr = 0.01, model_decay = 0.01):
        self.model = Sequential()
        self.model.add(Dense(10, input_dim=obs_dim+act_dim, activation='tanh'))
        self.model.add(Dense(8, activation='tanh'))
        self.model.add(Dense(obs_dim, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=model_lr, decay=model_decay))
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.train_freq  = train_freq
        self.memory = pd.DataFrame()
        self.last_memory_train = 0
        self.model_loss = 100
        self.control_points = control_points



    def observe(self, df):
        self.memory= self.memory.append(df)
        assert(df.shape[1]==self.obs_dim*2+self.act_dim)
        if (len(self.memory)-self.last_memory_train) > self.train_freq:
            X = self.memory.loc[:, range(self.act_dim + self.obs_dim)]
            y = self.memory.loc[:, range(self.act_dim + self.obs_dim, self.act_dim + self.obs_dim*2)]
            hist = self.model.fit(X, y, verbose=0)
            self.last_memory_train = len(self.memory)
            self.model_loss = hist.history["loss"]
            print("memory shape", self.memory.shape, "score: ", self.model_loss)

    def act(self, state):
        if len(self.memory)<self.train_freq+10: ##first actions are taken randomly
            action = env.action_space.sample()
        else:
            action = self.choose_action(state)
        return action

    def choose_action(self, state, strategies = 6, horizon = 3):
        ## to be redone
        df_actions = pd.DataFrame(np.random.randint(2, size=(strategies, horizon)))
        df_actions.iloc[:int(strategies/2),0] = 0
        df_actions.iloc[int(strategies/2):,0] = 1

        future_state = state.copy()

        df_future_state = pd.DataFrame([future_state] * strategies)
        for step in range(horizon):
            df_input = df_future_state.copy()
            df_input["action"] = df_actions[step]
            df_future_state = pd.DataFrame((self.model.predict(df_input)))
        # print("state", state)
       # print(df_future_state)
        # print(df_actions)
        obj_function =self.evaluate_obj(df_future_state, self.control_points)
        best_strategy = obj_function.idxmin()
        #best_strategy = abs(df_future_state[2]).idxmin()
        best_action = df_actions.iloc[best_strategy,0]
        # print("best action",best_action)
        return best_action

    @staticmethod
    def evaluate_obj(df, dict_obj):
        df["sum"] = 0
        for item in dict_obj:
            df["sum"] += abs(df.iloc[:,item]-dict_obj[item])
        return df["sum"]

if __name__ == '__main__':
    env = gym.make('Acrobot-v1')
   # agent_2 = agent()
   ## env = gym.make('Pendulum-v0')
    agent_2 = agent(obs_dim = 6, act_dim = 1, control_points = {0:1, 1:0}, train_freq = 100, model_lr = 0.01, model_decay = 0.01)


    def run_episode(render = False):
        state = env.reset()
        for t in range(2500):

            action = agent_2.act(state)

            observation, reward, done, info = env.step(action)
            if render == True:
                env.render()
            df_observation = pd.DataFrame([list(state) + [action] + list(observation)])
            agent_2.observe(df_observation)

            try:
                a = observation.shape[1]
                observation = observation[:,0]
            except:
                pass

            state = observation.copy() ###different f
            #print(t)
            if done:
                return t


    max_episodes = 300
    for episode in range(max_episodes):
        t = run_episode()
        print("Episode finished after {} timesteps".format(t + 1))
        if t < 498:
            print("finished at episonde", episode)
            break

    run_episode(render=True)



    # state = env.reset()
    # action = agent_2.act(state)
    # buba = agent_2.choose_action(state)
    # for t in range(250):
    #     print("time:", t)
    #     action = agent_2.act(state)
    #     print("action:",action)
    #     agent_2.choose_action(state)
    #     observation, reward, done, info = env.step(action)
    #     print("observation:", observation)
    #     df_observation = pd.DataFrame([list(state) + [action] + list(observation[:,0])])
    #     agent_2.observe(df_observation)
    #     print("observation:", observation[:,0])
    #     state = observation[:,0].copy()
    #     print("state:", state)