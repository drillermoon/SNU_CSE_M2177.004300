import numpy as np
import matplotlib.pyplot as plt

coef = 1.0  # lambda
numofsample = 1000000  # the number of samples


def exponential_cdf(x):  # cdf of exponential distribution
    return 1 - np.exp(-1 * coef * x)


def exponential_cdf_inverse(x):  # the inverse of the cdf
    return -1 / coef * np.log(1 - x)


# pick points from [0, 1] uniformly
sample = np.random.uniform(0, 1, numofsample)

x = np.linspace(0, 15, 501);

# plot the analytic cdf of exponential distribution
plt.plot(x, exponential_cdf(x))

# plot a normalized histogram of the samples
plt.hist(exponential_cdf_inverse(sample), bins=500, density=True)

plt.xlabel("value of X")
plt.ylabel("probability")

plt.show()  # draw the plots