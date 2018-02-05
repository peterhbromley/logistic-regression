#logistic-regression.py

# Python implementation of logistic regression

import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


# weighted probability of returning True
def weighted_flip(prob):
    return random.random() < prob

# general logistic function
def sigmoid(z):
    return 1/(1+np.exp(-z))

# log likelihood calculation given a dataset and a weight vector
def log_likelihood(x, y, theta):
    lm = np.dot(x, theta)
    total = -np.sum(y * lm - np.log(1 + np.exp(lm)))
    return total

# gradient descent weight update step
def update_weights(x, y, theta, lr):
    preds = sigmoid(np.dot(x, theta))
    update = lr * np.dot(x.transpose(), y - preds)
    return theta + update

# dataset generation given number of pts and target theta
def generate_data(nb_pts, theta):
    x = np.ones((nb_pts, 3))      # x[0] all ones for bias
    y = np.ones(nb_pts)
    for i in range(x.shape[0]):
        x[i][1] = random.uniform(-5, 5)
        x[i][2] = random.uniform(-5, 5)
        prob = sigmoid(np.dot(x[i], theta))
        y[i] = float(weighted_flip(prob))
    return x, y


# run logistic regression on the dataset, can specify learning rate, number of
# iterations, whether or not to save all of the thetas each iteration in an array,
# whether to print log likelihood.
#   Note:
#       save thetas if you want to run animation.
def run(x, y, lr=0.0001, nb_iter=50000, save_thetas=False, print_ll=False):

    theta = np.array([random.uniform(-5, 5) for i in range(x.shape[1])])
    saved_thetas = [theta]

    for i in range(nb_iter):
        theta = update_weights(x, y, theta, lr)

        if print_ll:
            if i % (nb_iter * 0.05) == 0:
                print(log_likelihood(x, y, theta))

        if save_thetas:
            saved_thetas.append(theta)


    if save_thetas:
        return saved_thetas
    else:
        return theta


# --------------------------------------
# Run a model:
theta_given = np.array([0, -1, 1])
x, y = generate_data(10000, theta_given)
saved_thetas = run(x, y, lr=0.00001, save_thetas=True, print_ll=True)


# Figures:
#---------------------------------------
# Plot dataset 2D:
plt.scatter(x[:, 1], x[:, 2], c = y, label=y)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Dataset Plotted in Two Dimensions")
plt.show()

# Plot dataset 3D:
fig = plt.figure()
ax = fig.gca(projection='3d')
scat = ax.scatter(x[:,1], x[:,2], y, depthshade=False, c=y)
ax.view_init(azim=225)
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Classification')
plt.show()

# Plot sigmoid 3D:
fig = plt.figure()
ax = fig.gca(projection='3d')
x_surf = np.arange(-5, 5, 0.25)
y_surf = np.arange(-5, 5, 0.25)
x_surf, y_surf = np.meshgrid(x_surf, y_surf)
z_surf = 1/(1+np.exp(-(1*x_surf + 1*y_surf)))
plot = ax.plot_surface(x_surf, y_surf, z_surf, alpha=0.8)
plt.show()


# Animation update function
def update_surface(frm, saved_thetas, z_surf, plot, x, y):
    current_theta = saved_thetas[frm]
    z_surf = 1/(1+np.exp(-(current_theta[1]*x_surf + current_theta[2]*y_surf)))
    ax.clear()
    plot = ax.plot_surface(x_surf, y_surf, z_surf, alpha=0.8)
    scat = ax.scatter(x[:,1], x[:,2], y, depthshade=False, c=y)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Classification')
    ax.set_title('3D Visualization of the Logistic Function During Logistic Regression')
    return plot


# Run animation
fig = plt.figure()
ax = fig.gca(projection='3d')
x_surf = np.arange(-5, 5, 0.25)
y_surf = np.arange(-5, 5, 0.25)
x_surf, y_surf = np.meshgrid(x_surf, y_surf)
z_surf = 1/(1+np.exp(-(saved_thetas[0][1]*x_surf + saved_thetas[0][1]*y_surf)))
ax.view_init(azim=225)
scat = ax.scatter(x[:,1], x[:,2], y, depthshade=False, c=y)
plot = ax.plot_surface(x_surf, y_surf, z_surf, alpha=0.8)
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Classification')
ax.set_title('3D Visualization of the Logistic Function During Logistic Regression')
anim = animation.FuncAnimation(fig, update_surface, fargs=(saved_thetas, z_surf, plot, x, y), repeat=True, interval=2)
plt.show()
