

#libro utilizado Data Science from Scratch: First Principles with Python, Joel Grus, 2015.


import pandas as pd
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math





dataset = pd.read_csv('dataset.csv')

def get_x(dataset):
    x = dict()
    for index, row in dataset.iterrows():
        NS = row['NS']
        MC = row['surface_total_in_m2']
        CA = row['rooms']
        x[index] = [1,NS,CA, MC]
    return x

def get_y(dataset):
    y = dict()
    for index, row in dataset.iterrows():
        y_i = row['price']
        y[index] = y_i
    return y

x = get_x(dataset)
y = get_y(dataset)




NS = dataset['NS']
MC = dataset['surface_total_in_m2']
CA = dataset['rooms']
VP = dataset['price']

def plot_model(x,y,z):
    threedee = plt.figure().gca(projection='3d')
    threedee.scatter(x,y,z)
    plt.show()
plot_model(NS,CA,MC)

dataset.plot.scatter(x="rooms", y="price")
plt.show()



def dot(v, w):
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

def predict(x_i, beta):
    return np.dot(x_i, beta)

def error(x_i, y_i, beta):
    return y_i - predict(x_i, beta)




def squared_error(x_i, y_i, beta):
    return error(x_i, y_i, beta)**2

def squared_error_gradient(x_i, y_i, beta):
    """the gradient (with respect to beta)
    corresponding to the ith squared error term"""
    return [-2 * x_ij * error(x_i, y_i, beta) for x_ij in x_i]

def in_random_order(data):
    indexes = [i for i, _ in enumerate(data)] # create a list of indexes
    random.shuffle(indexes)
    for i in indexes:
        yield data[i]

def minimize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01):
    data = zip(x.values(), y.values())
    theta = theta_0
    alpha = alpha_0
    min_theta, min_value = None, float("inf")
    iterations_with_no_improvement = 0
    
    while iterations_with_no_improvement < 100:
        value = sum(target_fn(x_i, y_i, theta) for x_i, y_i in data)
        if value < min_value:
            # if we've found a new minimum, remember it
            # and go back to the original step size
            min_theta, min_value = theta, value
            iterations_with_no_improvement = 0
            alpha = alpha_0
        else:
            iterations_with_no_improvement += 1
            alpha *= 0.9
        for x_i, y_i in in_random_order(data):
            gradient_i = gradient_fn(x_i, y_i, theta)
            theta = vector_subtract(theta, scalar_multiply(alpha, gradient_i))
    return min_theta




def estimate_beta(x, y):
    beta_initial = [random.random() for x_i in x[0]]
    return minimize_stochastic(squared_error, squared_error_gradient, x, y, beta_initial, 0.001)

random.seed(0)
beta = estimate_beta(x, y)
print("BETA = ", beta)




def least_squares_fit(x, y):
    beta = correlation(x, y) * standard_deviation(y) / standard_deviation(x)
    alpha = mean(y) - beta * mean(x)
    return alpha, beta

def total_sum_of_squares(y):
    return sum(v ** 2 for v in de_mean(y))

def multiple_r_squared(x, y, beta):
    sum_of_squared_errors = sum(error(x_i, y_i, beta) ** 2 for x_i, y_i in zip(x, y))
    return 1.0 - sum_of_squared_errors / total_sum_of_squares(y)

def vector_subtract(v, w):
    return [v_i - w_i for v_i, w_i in zip(v,w)]

def scalar_multiply(c, v):
    return [c * v_i for v_i in v]

def correlation(x, y):
    stdev_x = standard_deviation(x)
    stdev_y = standard_deviation(y)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(x, y) / stdev_x / stdev_y
    else:
        return 0

def standard_deviation(x):
    return math.sqrt(variance(x))

def variance(x):
    n = len(x)
    deviations = de_mean(x)
    return sum_of_squares(deviations) / (n - 1)

def de_mean(x):
    x_bar = mean(x)
    return [x_i - x_bar for x_i in x]

def mean(x):
    return sum(x) / len(x)

def sum_of_squares(v):
    return sum(v_i ** 2 for v_i in v)

def covariance(x, y):
    n = len(x)
    return dot(de_mean(x), de_mean(y)) / (n - 1)

def multiple_r_squared(x, y, beta):
    sum_of_squared_errors = sum(error(x_i, y_i, beta) ** 2 for x_i, y_i in zip(x, y))
    return 1.0 - sum_of_squared_errors / total_sum_of_squares(y)




print("R-SCUARED = ", multiple_r_squared(x, y, beta))




