import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from sklearn.model_selection import validation_curve
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


memory = pd.read_csv('out.csv')


#estimator = RandomForestRegressor(n_estimators=80, random_state=0)
estimator = MLPRegressor(solver='adam', alpha=1e-5,  hidden_layer_sizes=(5,5,3), max_iter=2000)
#clf = Ridge()

obs_dim = 6
act_dim = 1
model_lr = 0.01
model_decay = 0.01
#
# model = Sequential()
# model.add(Dense(10, input_dim=obs_dim + act_dim, activation='tanh'))
# model.add(Dense(8, activation='tanh'))
# model.add(Dense(obs_dim + 1, activation='linear'))
# model.compile(loss='mse', optimizer=Adam(lr=model_lr, decay=model_decay))

X = memory.iloc[:, range(act_dim + obs_dim)].copy()

#y = memory.iloc[:, range(act_dim + obs_dim, act_dim + obs_dim * 2 + 1)]
y = memory.iloc[:, range(act_dim + obs_dim+4, act_dim + obs_dim + 5)]
y = np.array(y)
y = y.reshape((len(y),))
#
#
#
# y = np.arccos(y)
# y = y-np.roll(y,1)
#

scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.reshape(-1,1))
y = y.reshape(len(y),)


hist = estimator.fit(X, y)

y_pred = estimator.predict(X)

X[:,6] = np.random.randint(3, size=(len(X),))

y_pred_2 = estimator.predict(X)

a = y - y_pred
b = y - y_pred_2

plt.plot(b)
plt.plot(a)



#
# param_range = np.logspace(-9, 5, 8)
# train_scores, test_scores = validation_curve(estimator, X, y, "alpha", param_range, cv=5)
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)
#
# plt.title("Validation Curve with SVM")
# plt.xlabel(r"$\gamma$")
# plt.ylabel("Score")
# plt.ylim(0.0, 1.1)
# lw = 2
# plt.semilogx(param_range, train_scores_mean, label="Training score",
#              color="darkorange", lw=lw)
# plt.fill_between(param_range, train_scores_mean - train_scores_std,
#                  train_scores_mean + train_scores_std, alpha=0.2,
#                  color="darkorange", lw=lw)
# plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
#              color="navy", lw=lw)
# plt.fill_between(param_range, test_scores_mean - test_scores_std,
#                  test_scores_mean + test_scores_std, alpha=0.2,
#                  color="navy", lw=lw)
# plt.legend(loc="best")
# plt.show()
# #
# def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
#                         n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
#     """
#     Generate a simple plot of the test and training learning curve.
#
#     Parameters
#     ----------
#     estimator : object type that implements the "fit" and "predict" methods
#         An object of that type which is cloned for each validation.
#
#     title : string
#         Title for the chart.
#
#     X : array-like, shape (n_samples, n_features)
#         Training vector, where n_samples is the number of samples and
#         n_features is the number of features.
#
#     y : array-like, shape (n_samples) or (n_samples, n_features), optional
#         Target relative to X for classification or regression;
#         None for unsupervised learning.
#
#     ylim : tuple, shape (ymin, ymax), optional
#         Defines minimum and maximum yvalues plotted.
#
#     cv : int, cross-validation generator or an iterable, optional
#         Determines the cross-validation splitting strategy.
#         Possible inputs for cv are:
#           - None, to use the default 3-fold cross-validation,
#           - integer, to specify the number of folds.
#           - :term:`CV splitter`,
#           - An iterable yielding (train, test) splits as arrays of indices.
#
#         For integer/None inputs, if ``y`` is binary or multiclass,
#         :class:`StratifiedKFold` used. If the estimator is not a classifier
#         or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
#
#         Refer :ref:`User Guide <cross_validation>` for the various
#         cross-validators that can be used here.
#
#     n_jobs : int or None, optional (default=None)
#         Number of jobs to run in parallel.
#         ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
#         ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
#         for more details.
#
#     train_sizes : array-like, shape (n_ticks,), dtype float or int
#         Relative or absolute numbers of training examples that will be used to
#         generate the learning curve. If the dtype is float, it is regarded as a
#         fraction of the maximum size of the training set (that is determined
#         by the selected validation method), i.e. it has to be within (0, 1].
#         Otherwise it is interpreted as absolute sizes of the training sets.
#         Note that for classification the number of samples usually have to
#         be big enough to contain at least one sample from each class.
#         (default: np.linspace(0.1, 1.0, 5))
#     """
#     plt.figure()
#     plt.title(title)
#     if ylim is not None:
#         plt.ylim(*ylim)
#     plt.xlabel("Training examples")
#     plt.ylabel("Score")
#     train_sizes, train_scores, test_scores = learning_curve(
#         estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
#     test_scores_std = np.std(test_scores, axis=1)
#     plt.grid()
#
#     plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                      train_scores_mean + train_scores_std, alpha=0.1,
#                      color="r")
#     plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                      test_scores_mean + test_scores_std, alpha=0.1, color="g")
#     plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
#              label="Training score")
#     plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
#              label="Cross-validation score")
#
#     plt.legend(loc="best")
#     return plt
#
#
# # digits = load_digits()
# # X, y = digits.data, digits.target
# #
#
# title = "Learning Curves (Naive Bayes)"
# # Cross validation with 100 iterations to get smoother mean test and train
# # score curves, each time with 20% data randomly selected as a validation set.
# cv = 5#ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
#
#
# plot_learning_curve(estimator, title, X, y, ylim=(0, 1.01), cv=cv, n_jobs=4)
#
# #title = r"Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
# # SVC is more expensive so we do a lower number of CV iterations:
# #cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
# #estimator = SVC(gamma=0.001)
# #plot_learning_curve(estimator, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)
#
# plt.show()