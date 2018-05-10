import numpy as np
import matplotlib.pyplot as plt

# set hyperparameter
coeff = 0.1
n = 1000
d = 100
step_size = 0.01
epsilon = 1.0E-6

# fix the random seed
np.random.seed(1337)

# make training data
X = np.vstack([np.random.normal(0.1, 1, (n//2, d)),
               np.random.normal(-0.1, 1, (n//2, d))])
Y = np.hstack([np.ones(n//2), -1*np.ones(n//2)])

# initialise parameter
w0 = np.random.normal(0, 1, d)

fnt_val = []
accuracy  = []
loss_prev = 0

while True:
    correct = 0
    grad = np.zeros(d)
    loss = 0

    # calculate the gradient and the value of the loss function
    for i in range(n):
        x = X[i, :]
        y = Y[i]
        if y*np.matmul(w0, x)<1:
            grad += -y*x
            loss += 1-y*np.matmul(w0, x)
        if y*np.matmul(w0, x)>0:
            correct += 1
    grad /= n
    grad += coeff*w0

    loss += coeff/2*np.matmul(w0, w0)

    #update the parameter
    w0 -= step_size*grad

    print(np.abs(loss-loss_prev))
    #exit condition
    if np.abs(loss-loss_prev)<epsilon:
        break

    fnt_val.append(loss)
    accuracy.append(correct/n*100)
    loss_prev = loss

#draw graphs
fig = plt.figure()

plt1 = fig.add_subplot(1, 2, 1)
plt1.set_title("the value of loss function")
plt1.set_xlabel("num of iteration")
plt1.set_ylabel("val of loss funtion")
plt1.set_ylim(0, 5000)
plt1.set_yticks(np.linspace(0, 5000, 11))

plt2 = fig.add_subplot(1, 2, 2)
plt2.set_title("accuracy")
plt2.set_xlabel("num of iteration")
plt2.set_ylabel("accuracy(%)")
plt2.set_ylim(0, 100)
plt2.set_yticks(np.linspace(0, 100, 11))

plt1.plot(fnt_val)
plt2.plot(accuracy)
plt.show()