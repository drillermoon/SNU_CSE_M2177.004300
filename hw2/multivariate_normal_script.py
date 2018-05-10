import numpy as np
import matplotlib.pyplot as plt

#fix the random seed
np.random.seed(1337)

# generate two normal random variable, which are independent
X = np.random.standard_normal([2, 10000]);


E = np.array([[np.sqrt(1.9), 0], [0, np.sqrt(0.1)]])
P = np.array([[1/np.sqrt(2), 1/np.sqrt(2)], [1/np.sqrt(2), -1/np.sqrt(2)]])

# transform the variables
Z = np.matmul(E, X)
Z = np.matmul(P, Z)

# generate the multivariate random variable withe the given mean and covariance
Y1, Y2 = np.random.multivariate_normal([0, 0], [[1, 0.9],[0.9, 1]], 10000).T


# draw scatter plots
fig = plt.figure()

plt1 = fig.add_subplot(1, 3, 1)
plt2 = fig.add_subplot(1, 3, 2)
plt3 = fig.add_subplot(1, 3, 3)

plt1.scatter(X[0, :], X[1, :], s=0.1)
plt1.set_title("standard normal")

plt2.scatter(Z[0, :], Z[1, :], s=0.1)
plt2.set_title("my multivariate")

plt3.scatter(Y1, Y2, s=0.1)
plt3.set_title("numpy's multivariate")

plt.show()
