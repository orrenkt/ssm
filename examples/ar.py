import autograd.numpy as np
import autograd.numpy.random as npr
npr.seed(0)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set_style("white")
sns.set_context("talk")
color_names = ["windows blue",
               "red",
               "amber",
               "faded green",
               "dusty purple",
               "orange",
               "clay",
               "pink",
               "greyish",
               "mint",
               "light cyan",
               "steel blue",
               "forest green",
               "pastel purple",
               "salmon",
               "dark brown"]
colors = sns.xkcd_palette(color_names)
from ssm import SLDS
from ssm.util import random_rotation

# Set the parameters of the SLDS
T = 200   # number of time bins
K = 1      # number of discrete states
D = 2      # number of latent dimensions
N = 20     # number of observed dimensions
M = 0      # number of inputs
lags = 3

# Make an LDS with somewhat interesting dynamics parameters
true_lds = SLDS(N, K, D, M=M, dynamics="higher_order", emissions="gaussian",
                dynamics_kwargs={'lags':lags})
A0 = .7 * random_rotation(D, theta=np.pi/20)
S = np.arange(1, D+1)
R = np.linalg.svd(npr.randn(D, D))[0] * S
b = npr.randn(D)
true_lds.dynamics.bs[0] = b
true_lds.emissions.Fs[0] = 0.0 * true_lds.emissions.Fs[0]

# Sample
# us = 0.2 * npr.choice([0,1], (T, M), replace=True)
us = 0.2 * npr.randn(T, M)
z, x, y = true_lds.sample(T) #, input=us)
print("Fitting LDS with Laplace EM")
lds = SLDS(N, K, D, M=M, dynamics="higher_order", emissions="gaussian", F_zero_flag=True,
           dynamics_kwargs={'lags':lags})
lds.emissions.Fs[0] = 0.0 * true_lds.emissions.Fs[0]
lds.initialize(y, inputs=us)
kwargs = {}
q_lem_elbos, q_lem = lds.fit(y, inputs=us, method="laplace_em", variational_posterior="structured_meanfield_banded",
                             num_iters=20, initialize=False, continuous_optimizer='newton')  #lbfgs newton
print(type(q_lem))

q_lem_x = q_lem.mean_continuous_states[0]
from sklearn.linear_model import LinearRegression
lr = LinearRegression(fit_intercept=True)
lr.fit(q_lem_x, x)
q_lem_x_trans = lr.predict(q_lem_x)
# Smooth the data under the variational posterior
q_lem_y = lds.smooth(q_lem_x, y, input=us)

plt.figure(figsize=(8,4))
plt.plot(x + 4 * np.arange(D), '-k')
for d in range(D):
    plt.plot(q_lem_x_trans[:,d] + 4 * d, '--', color=colors[2], label="Laplace-EM" if d==0 else None)
plt.ylabel("$x$")
plt.xlim((0,T))
plt.legend()

# Plot the smoothed observations
plt.figure(figsize=(8,4))
for n in range(N):
    plt.plot(y[:, n] + 4 * n, '-k', label="True" if n == 0 else None)
    plt.plot(q_lem_y[:, n] + 4 * n, '--',  color=colors[2], label="Laplace-EM" if n == 0 else None)
plt.legend()
plt.xlabel("time")
plt.xlim((0,T))



# lds2 = SLDS(N, K, D, M=M, emissions="gaussian", F_zero_flag=True)
# lds2.emissions.Fs[0] = 0.0 * true_lds.emissions.Fs[0]
# lds2.initialize(y, inputs=us)
# kwargs = {}
# q_lem_elbos2, q_lem2 = lds2.fit(y, inputs=us, method="laplace_em", variational_posterior="structured_meanfield",
#                              num_iters=20, initialize=False, continuous_optimizer='newton')  #lbfgs newton
