import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from sklearn.model_selection import validation_curve
from sklearn.linear_model import Ridge

memory = pd.read_csv('out.csv')


# obs_dim = 4
# act_dim = 1
# model_lr = 0.01
# model_decay = 0.01
#
# model = Sequential()
# model.add(Dense(10, input_dim=obs_dim + act_dim, activation='tanh'))
# model.add(Dense(8, activation='tanh'))
# model.add(Dense(obs_dim + 1, activation='linear'))
# model.compile(loss='mse', optimizer=Adam(lr=model_lr, decay=model_decay))
#
# X = memory.iloc[:, range(act_dim + obs_dim)]
# #y = memory.iloc[:, range(act_dim + obs_dim, act_dim + obs_dim * 2 + 1)]
# y = memory.iloc[:, range(act_dim + obs_dim, act_dim + obs_dim + 1)]
# y = np.array(y)
# y = y.reshape((len(y),))
# #hist = model.fit(X, y, verbose=0)

np.random.seed(0)
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target
indices = np.arange(y.shape[0])
np.random.shuffle(indices)
X, y = X[indices], y[indices]

param_range = np.logspace(-7, 3, 3)
train_scores, test_scores = validation_curve(Ridge(), X, y, "alpha", param_range, cv=5)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with SVM")
plt.xlabel(r"$\gamma$")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()