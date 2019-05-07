
import random
import gym
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
from sklearn.preprocessing import normalize


error_store = []

class agent():

    def __init__(self, obs_dim = 4, act_dim = 1, train_freq = 100, model_lr = 0.01, model_decay = 0.01, from_file=False):
        self.model = Sequential()
        self.model.add(Dense(10, input_dim=obs_dim+act_dim, activation='tanh'))
        self.model.add(Dense(8, activation='tanh'))
        self.model.add(Dense(obs_dim+1, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=model_lr, decay=model_decay))
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.train_freq  = train_freq
        self.memory = pd.DataFrame()
        self.last_memory_train = 0
        self.model_loss = 100

        if from_file == True:
            self.memory = pd.read_csv('out.csv')
            self.memory.columns = range(14)

        self.error_threshold = 0.4


    def process_info(self, df):
        assert(df.shape[1]==self.obs_dim*2+self.act_dim + 1)
        X = self.memory.iloc[:, range(self.act_dim + self.obs_dim)]
        y = self.memory.iloc[:, range(self.act_dim + self.obs_dim, self.act_dim + self.obs_dim*2 + 1)]
        hist = self.model.fit(X, y, verbose=0)
        self.last_memory_train = len(self.memory)
        self.model_loss = hist.history["loss"]
        print("memory shape", self.memory.shape, "score: ", self.model_loss)
        try:
            print(np.average(error_store[:-30]))
        except:
            pass


    def live(self, state, env):
        if len(self.memory)<self.train_freq+1: ##first actions are taken randomly
            action = env.action_space.sample()
            expectation = None
        else:
            action, expectation= self.choose_action(state)

        observation, reward, done, info = env.step(action)
        df_info = pd.DataFrame([list(state) + [action] + list(observation) + [reward]])

        if expectation is None:
            self.memory = self.memory.append(df_info)
        else:
            error = self.mean_absolute_percentage_error(pd.DataFrame([list(observation) + [reward]]), expectation[:, :])
            error_store.append(np.amax(error))
            if np.amax(error)>self.error_threshold:
                self.memory = self.memory.append(df_info)
            else:
                pass

        if (len(self.memory) - self.last_memory_train) > self.train_freq:
            self.process_info(df_info)


        return observation, reward, done, info


    def choose_action(self, state, strategies = 6, horizon = 6):
        ## to be redone
        df_actions = pd.DataFrame(np.random.randint(2, size=(strategies, horizon)))
        # df_actions.iloc[:int(strategies/3),0] = 0
        # df_actions.iloc[int(strategies/3):,0] = 1

        future_state = state.copy()

        df_future_state = pd.DataFrame([future_state] * strategies)
        df_total_reward = np.array(pd.DataFrame([0] * strategies).iloc[:,0])
        df_total_uncertainty = np.array(pd.DataFrame([0] * strategies).iloc[:,0])
        for step in range(horizon):
            df_input = df_future_state.copy()
            df_input["action"] = df_actions[step]

            # a = np.array([[2,3,5,6,7,6,7],[4,5,4,3,6,6,8]])
            # b = np.array([[6,3],[5,5],[8,8]])
            a = np.array(df_input)
            d = np.array(self.memory.iloc[:,:7])
            c = self.calc_distance(a,d)
            uncertainty = pd.DataFrame(c.transpose())

            output = pd.DataFrame((self.model.predict(df_input)))
            df_future_state = output.iloc[:,:-1]
            df_total_reward = df_total_reward  + np.array(output.iloc[:,-1])

            df_total_uncertainty = df_total_uncertainty +np.array(uncertainty).reshape(1,-1)
            ### at each step calculate distances in memory and prediction.

        #decide to explore or exploit.
        # if the point is in reliable space set expectation

        # print("state", state)
       # print(df_future_state)
        # print(df_actions)
        #obj_function =self.evaluate_obj(df_future_state, self.control_points)
        obj_function = 0*df_total_reward + df_total_uncertainty
        #obj_function = 0 * df_total_reward - tot_confidence.iloc[:,0]
        best_strategy = obj_function.argmax()
        #best_strategy = abs(df_future_state[2]).idxmin()
        best_action = df_actions.iloc[best_strategy,0]

        input = pd.DataFrame([list(state) + [best_action]])
        expectation = self.model.predict(input)
        # print("best action",best_action)



        # distances, indices = self.nbrs.kneighbors(input)
        # indices = indices[distances < self.max_distance_nbrs]
        # try:
        #     confidence = np.mean(self.memory.iloc[indices[0],-1])
        # except:
        #     confidence = 0
        #print("confidence", confidence, "distance", distances)
        #print(obj_function)

        return best_action, expectation

    @staticmethod
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.abs(y_true - y_pred) / np.maximum(np.maximum(abs(y_true), abs(y_pred)),0.00001)

    @staticmethod
    def calc_distance(y, df_memory):
        distances = []
        for item in y:
        #y=preprocessing.scale(df_memory.append(y))
        #dist = np.linalg.norm(y[:-1,:]-y[-1,:], axis=1)
            dist = np.linalg.norm( df_memory-item, axis=1)
            distances.append(dist.min())
        #res = normalize(np.array(distances).reshape(1,-1))
        res = np.array(distances).reshape(1, -1)
        return res


if __name__ == '__main__':
    env = gym.make('Acrobot-v1')
   # agent_2 = agent()
   ## env = gym.make('Pendulum-v0')
    agent_2 = agent(obs_dim = 6, act_dim = 1,  train_freq = 100, model_lr = 0.01, model_decay = 0.01, from_file=True)


    def run_episode(render = False):
        state = env.reset()
        for t in range(2500):

            observation, reward, done, info= agent_2.live(state, env)

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


    max_episodes = 30
    best_t = 500
    for episode in range(max_episodes):
        t = run_episode()
        print("Episode", episode, "finished after {} timesteps".format(t + 1), "best time:", best_t)
        if t < best_t:
            best_t = t
        # print("finished at episonde", episode)

    agent_2.memory.to_csv('out.csv', index=False)

    run_episode(render=True)
