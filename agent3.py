
import random
import gym
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import time
from sklearn.neighbors import NearestNeighbors
import math

target = 1000
timescale = 5

#
# self.max_points_space = 1000
# var_point = int(np.power(self.max_points_space, 1/obs_dim))
#self.confidence_space = np.zeros(tuple(var_point for i in range(obs_dim+act_dim)))
# self.value_space = np.zeros(tuple(var_point for i in range(obs_dim)))

class agent():

    def __init__(self, obs_dim = 4, act_dim = 1, control_points = {2:0.00}, train_freq = 100, model_lr = 0.01, model_decay = 0.01):
        self.model = Sequential()
        self.model.add(Dense(10, input_dim=obs_dim+act_dim, activation='tanh'))
        self.model.add(Dense(8, activation='tanh'))
        self.model.add(Dense(obs_dim+2, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=model_lr, decay=model_decay))
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.train_freq  = train_freq
        self.memory = pd.DataFrame()
        self.last_memory_train = 0
        self.model_loss = 100
        self.control_points = control_points
        #self.reward = 0
        #self.max_points_space = 1000
        #var_point = int(np.power(self.max_points_space, 1 / (obs_dim+ act_dim)))
        #self.confidence_space = np.zeros(tuple(var_point for i in range(obs_dim + act_dim)))
        self.error_threshold = 0.4
        self.nbrs =  NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
        self.max_distance_nbrs = 0.3


    def observe(self, df, df_observation_, confidence, expectation = None):

        if expectation is None:
            df["confidence"] = 0
        elif confidence>0.2:
            error = self.mean_absolute_percentage_error(df_observation_, expectation[:,:-1])
            #print("error", error, "obs", df_observation_, "exp",df_expectation[:,:-1] )
            if np.amax(error) > self.error_threshold:
                df["confidence"] = -1
            else:
                df["confidence"] = 1
        else:
            error = self.mean_absolute_percentage_error(df_observation_, expectation[:, :-1])
            # print("error", error, "obs", df_observation_, "exp",df_expectation[:,:-1] )
            if np.amax(error) > self.error_threshold:
                df["confidence"] = 0
            else:
                df["confidence"] = 0.3
        # if expectation is not None:
        #     print("confidence",expectation[:, -1], "error", np.amax(error))
        self.memory = self.memory.append(df)
        assert(df.shape[1]==self.obs_dim*2+self.act_dim + 2)
        if (len(self.memory)-self.last_memory_train) > self.train_freq:
            X = self.memory.iloc[:, range(self.act_dim + self.obs_dim)]
            y = self.memory.iloc[:, range(self.act_dim + self.obs_dim, self.act_dim + self.obs_dim*2 + 2)]
            hist = self.model.fit(X, y, verbose=0)
            self.last_memory_train = len(self.memory)
            self.model_loss = hist.history["loss"]
            print("memory shape", self.memory.shape, "score: ", self.model_loss)
            self.nbrs.fit(X)






    def act(self, state, env):
        if len(self.memory)<self.train_freq+1: ##first actions are taken randomly
            action = env.action_space.sample()
            expectation = None
            confidence = 0
        else:
            action, expectation, confidence = self.choose_action(state)
        #print("expect", expectation)
        observation, reward, done, info = env.step(action)
        df_observation = pd.DataFrame([list(state) + [action] + list(observation) + [reward]])
        df_observation_ = pd.DataFrame([list(observation) + [reward]])

        self.observe(df_observation, df_observation_, confidence, expectation)
        # self.reward += reward
        return observation, reward, done, info


    def choose_action(self, state, strategies = 6, horizon = 6):
        ## to be redone
        df_actions = pd.DataFrame(np.random.randint(2, size=(strategies, horizon)))
        df_actions.iloc[:int(strategies/2),0] = 0
        df_actions.iloc[int(strategies/2):,0] = 1

        future_state = state.copy()

        df_future_state = pd.DataFrame([future_state] * strategies)
        df_total_reward = pd.DataFrame([0] * strategies).iloc[:,0]
        df_total_confidence = pd.DataFrame([0] * strategies).iloc[:,0]
        for step in range(horizon):
            df_input = df_future_state.copy()
            df_input["action"] = df_actions[step]
            output = pd.DataFrame((self.model.predict(df_input)))
            df_future_state = output.iloc[:,:-2]
            df_total_reward = df_total_reward  + output.iloc[:,-2]
            df_total_confidence = output.iloc[:,-1]

        # my_input = df_future_state.copy()
        # my_input["action"] = df_actions[0]
        # distances, indices = self.nbrs.kneighbors(my_input)
        # a = distances < self.max_distance_nbrs
        # m = agent_2.memory.iloc[indices[a], -1]
        # ind = np.cumsum(np.sum(a, axis=1))
        # tot_confidence = []
        # ind = list(ind)
        # ind = [0] + ind
        # for i in range(indices.shape[0]):
        #     val = np.mean(m.iloc[ind[i]:ind[i + 1]])
        #     if math.isnan(val):
        #         val = 0
        #     tot_confidence.append(val)
        #
        # tot_confidence = pd.DataFrame({"confidence": tot_confidence})


        # print("state", state)
       # print(df_future_state)
        # print(df_actions)
        #obj_function =self.evaluate_obj(df_future_state, self.control_points)
        obj_function = 0*df_total_reward - df_total_confidence
        #obj_function = 0 * df_total_reward - tot_confidence.iloc[:,0]
        best_strategy = obj_function.idxmax()
        #best_strategy = abs(df_future_state[2]).idxmin()
        best_action = df_actions.iloc[best_strategy,0]
        input = pd.DataFrame([list(state) + [best_action]])
        expectation = self.model.predict(input)
        # print("best action",best_action)

        #confidence = tot_confidence.iloc[best_strategy,0]
        confidence = df_total_confidence.iloc[best_strategy]

        # distances, indices = self.nbrs.kneighbors(input)
        # indices = indices[distances < self.max_distance_nbrs]
        # try:
        #     confidence = np.mean(self.memory.iloc[indices[0],-1])
        # except:
        #     confidence = 0
        #print("confidence", confidence, "distance", distances)
        #print(obj_function)

        return best_action, expectation, confidence

    @staticmethod
    def evaluate_obj(df, dict_obj):
        df["sum"] = 0
        for item in dict_obj:
            df["sum"] += abs(df.iloc[:,item]-dict_obj[item])
        return df["sum"]

    @staticmethod
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.abs(y_true - y_pred) / np.maximum(np.maximum(abs(y_true), abs(y_pred)),0.00001)



if __name__ == '__main__':
    env = gym.make('Acrobot-v1')
   # agent_2 = agent()
   ## env = gym.make('Pendulum-v0')
    agent_2 = agent(obs_dim = 6, act_dim = 1, control_points = {0:1, 1:0}, train_freq = 100, model_lr = 0.01, model_decay = 0.01)


    def run_episode(render = False):
        state = env.reset()
        for t in range(2500):

            observation, reward, done, info= agent_2.act(state, env)

            #observation, reward, done, info = env.step(action)
            if render == True:
                env.render()
            # df_observation = pd.DataFrame([list(state) + [action] + list(observation)])
            # agent_2.observe(df_observation)

            # try:
            #     a = observation.shape[1]
            #     observation = observation[:,0]
            # except:
            #     pass

            state = observation.copy() ###different f
            #print(t)
            if done:
                return t


    max_episodes = 5
    best_t = 500
    for episode in range(max_episodes):
        t = run_episode()
        print("Episode finished after {} timesteps".format(t + 1), "best time:", best_t)
        if t < best_t:
            best_t = t
        # print("finished at episonde", episode)


    run_episode(render=True)
    #
    #

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