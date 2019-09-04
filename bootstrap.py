#!/usr/bin/env python
# coding: utf-8

# In[481]:

from typing import List
import pandas as pd
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math


# In[484]:


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


# In[487]:


NS = dataset['NS']
MC = dataset['surface_total_in_m2']
CA = dataset['rooms']
VP = dataset['price']

def plot_model(x,y,z):
    threedee = plt.figure().gca(projection='3d')
    threedee.scatter(x,y,z)
    #threedee.scatter(0.7579544029403025, 0.420571580830845,0.25891675029296335)
    plt.show()
plot_model(NS,CA,MC)

dataset.plot.scatter(x="rooms", y="price")
plt.show()

# In[470]:


def dot(v, w):
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

def predict(x_i, beta):
    return np.dot(x_i, beta)

def error(x_i, y_i, beta):
    return y_i - predict(x_i, beta)


# In[472]:


def squared_error(x_i, y_i, beta):
    return error(x_i, y_i, beta)**2

def squared_error_gradient(x_i, y_i, beta):
    """the gradient (with respect to beta)
    corresponding to the ith squared error term"""
    return [-2 * x_ij * error(x_i, y_i, beta) for x_ij in x_i]

def in_random_order(data):
    """generator that returns the elements of data in random order"""
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
            # otherwise we're not improving, so try shrinking the step size
            iterations_with_no_improvement += 1
            alpha *= 0.9
            # and take a gradient step for each of the data points
        for x_i, y_i in in_random_order(data):
            gradient_i = gradient_fn(x_i, y_i, theta)
            theta = vector_subtract(theta, scalar_multiply(alpha, gradient_i))
    return min_theta


# In[473]:


def estimate_beta(x, y):
    beta_initial = [random.random() for x_i in x[0]]
    return minimize_stochastic(squared_error, squared_error_gradient, x, y, beta_initial, 0.001)

random.seed(0)
beta = estimate_beta(x, y)
print("BETA = ", beta)


# In[474]:


def least_squares_fit(x, y):
    beta = correlation(x, y) * standard_deviation(y) / standard_deviation(x)
    alpha = mean(y) - beta * mean(x)
    return alpha, beta

def total_sum_of_squares(y):
    return sum(v ** 2 for v in de_mean(y))

def multiple_r_squared(x, y, beta):
    sum_of_squared_errors = sum(error(x_i, y_i, beta) ** 2 for x_i, y_i in zip(x, y))
    return 1.0 - sum_of_squared_errors / total_sum_of_squares(y)


from typing import TypeVar, Callable

X = TypeVar('X')        # Generic type for data
Stat = TypeVar('Stat') # Generic type for "statistic"


def bootstrap_sample(data: List[X]) -> List[X]:
	"""randomly samples len(data) elements with replacement"""
	return [random.choice(data) for _ in data]
def bootstrap_statistic(data: List[X],
                        stats_fn: Callable[[List[X]], Stat],
                        num_samples: int) -> List[Stat]:
	"""evaluates stats_fn on num_samples bootstrap samples from data"""
	return [stats_fn(bootstrap_sample(data)) for _ in range(num_samples)]
# 101 points all very close to 100
close_to_100 = [99.5 + random.random() for _ in range(101)]

# 101 points, 50 of them near 0, 50 of them near 200
far_from_100 = ([99.5 + random.random()] +
                [random.random() for _ in range(50)] +
                [200 + random.random() for _ in range(50)])

#from scratch.statistics import median, standard_deviation

def sum_of_squares(v: Vector) -> float:
	return dot(v, v)

def mean(xs: List[float]) -> float:
	return sum(xs) / len(xs)


def de_mean(xs: List[float]) -> List[float]:
	x_bar = mean(xs)
	return [x - x_bar for x in xs]

def variance(xs: List[float]) -> float:
	assert len(xs) >= 2, "variance requires at least two elements"
	n = len(xs)
	deviations = de_mean(xs)
	return sum_of_squares(deviations) / (n - 1)

def standard_deviation(xs: List[float]) -> float:
	return math.sqrt(variance(xs))

def _median_odd(xs: List[float]) -> float:
	return sorted(xs)[len(xs) // 2]

def _median_even(xs: List[float]) -> float:
	sorted_xs = sorted(xs)
	hi_midpoint = len(xs) // 2
	return (sorted_xs[hi_midpoint - 1] + sorted_xs[hi_midpoint])/2

def median(v: List[float]) -> float:
	return _median_even(v) if len(v) % 2 == 0 else _median_odd(v)

medians_close = bootstrap_statistic(close_to_100, median, 100)

medians_far = bootstrap_statistic(far_from_100, median, 100)

assert standard_deviation(medians_close) < 1
assert standard_deviation(medians_far) > 90

from scratch.probability import normal_cdf
def p_value(beta_hat_j: float, sigma_hat_j: float) -> float:
    if beta_hat_j > 0:
        # if the coefficient is positive, we need to compute twice the
        # probability of seeing an even *larger* value
        return 2 * (1 - normal_cdf(beta_hat_j / sigma_hat_j))
    else:
        # otherwise twice the probability of seeing a *smaller* value
        return 2 * normal_cdf(beta_hat_j / sigma_hat_j)

assert p_value(30.58, 1.27)   < 0.001  # constant term
assert p_value(0.972, 0.103)  < 0.001  # num_friends
assert p_value(-1.865, 0.155) < 0.001  # work_hours
assert p_value(0.923, 1.249) > 0.4 # phd

# alpha is a *hyperparameter* controlling how harsh the penalty is
# sometimes it's called "lambda" but that already means something in Python
def ridge_penalty(beta: Vector, alpha: float) -> float:
    return alpha * dot(beta[1:], beta[1:])

def squared_error_ridge(x: Vector,
                        y: float,
                        beta: Vector,
                        alpha: float) -> float:
    """estimate error plus ridge penalty on beta"""
    return error(x, y, beta) ** 2 + ridge_penalty(beta, alpha)

from scratch.linear_algebra import add

def ridge_penalty_gradient(beta: Vector, alpha: float) -> Vector:
    """gradient of just the ridge penalty"""
    return [0.] + [2 * alpha * beta_j for beta_j in beta[1:]]



def sqerror_ridge_gradient(x: Vector,
                           y: float,
                           beta: Vector,
                           alpha: float) -> Vector:
    """
    the gradient corresponding to the ith squared error term
    including the ridge penalty
    """
    return add(sqerror_gradient(x, y, beta),
               ridge_penalty_gradient(beta, alpha))


from scratch.statistics import daily_minutes_good
from scratch.gradient_descent import gradient_step



learning_rate = 0.001

def least_squares_fit_ridge(xs: List[Vector],
                            ys: List[float],
                            alpha: float,
                            learning_rate: float,
                            num_steps: int,
                            batch_size: int = 1) -> Vector:
    # Start guess with mean
    guess = [random.random() for _ in xs[0]]

    for i in range(num_steps):
        for start in range(0, len(xs), batch_size):
            batch_xs = xs[start:start+batch_size]
            batch_ys = ys[start:start+batch_size]

            gradient = vector_mean([sqerror_ridge_gradient(x, y, guess, alpha)
                                    for x, y in zip(batch_xs, batch_ys)])
            guess = gradient_step(guess, gradient, -learning_rate)

    return guess

def lasso_penalty(beta, alpha):
	return alpha * sum(abs(beta_i) for beta_i in beta[1:])







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

def main():
    from scratch.statistics import daily_minutes_good
    from scratch.gradient_descent import gradient_step
    
    random.seed(0)
    # I used trial and error to choose niters and step_size.
    # This will run for a while.
    learning_rate = 0.001
    
    beta = least_squares_fit(inputs, daily_minutes_good, learning_rate, 5000, 25)
    assert 30.50 < beta[0] < 30.70  # constant
    assert  0.96 < beta[1] <  1.00  # num friends
    assert -1.89 < beta[2] < -1.85  # work hours per day
    assert  0.91 < beta[3] <  0.94  # has PhD
    
    assert 0.67 < multiple_r_squared(inputs, daily_minutes_good, beta) < 0.68
    
    from typing import Tuple
    
    import datetime
    
    def estimate_sample_beta(pairs: List[Tuple[Vector, float]]):
        x_sample = [x for x, _ in pairs]
        y_sample = [y for _, y in pairs]
        beta = least_squares_fit(x_sample, y_sample, learning_rate, 5000, 25)
        print("bootstrap sample", beta)
        return beta
    
    random.seed(0) # so that you get the same results as me
    
    # This will take a couple of minutes!
    bootstrap_betas = bootstrap_statistic(list(zip(inputs, daily_minutes_good)),
                                          estimate_sample_beta,
                                          100)
    
    bootstrap_standard_errors = [
        standard_deviation([beta[i] for beta in bootstrap_betas])
        for i in range(4)]
    
    print(bootstrap_standard_errors)
    
    # [1.272,    # constant term, actual error = 1.19
    #  0.103,    # num_friends,   actual error = 0.080
    #  0.155,    # work_hours,    actual error = 0.127
    #  1.249]    # phd,           actual error = 0.998
    
    random.seed(0)
    beta_0 = least_squares_fit_ridge(inputs, daily_minutes_good, 0.0,  # alpha
                                     learning_rate, 5000, 25)
    # [30.51, 0.97, -1.85, 0.91]
    assert 5 < dot(beta_0[1:], beta_0[1:]) < 6
    assert 0.67 < multiple_r_squared(inputs, daily_minutes_good, beta_0) < 0.69
    
    beta_0_1 = least_squares_fit_ridge(inputs, daily_minutes_good, 0.1,  # alpha
                                       learning_rate, 5000, 25)
    # [30.8, 0.95, -1.83, 0.54]
    assert 4 < dot(beta_0_1[1:], beta_0_1[1:]) < 5
    assert 0.67 < multiple_r_squared(inputs, daily_minutes_good, beta_0_1) < 0.69
    
    
    beta_1 = least_squares_fit_ridge(inputs, daily_minutes_good, 1,  # alpha
                                     learning_rate, 5000, 25)
    # [30.6, 0.90, -1.68, 0.10]
    assert 3 < dot(beta_1[1:], beta_1[1:]) < 4
    assert 0.67 < multiple_r_squared(inputs, daily_minutes_good, beta_1) < 0.69
    
    beta_10 = least_squares_fit_ridge(inputs, daily_minutes_good,10,  # alpha
                                      learning_rate, 5000, 25)
    # [28.3, 0.67, -0.90, -0.01]
    assert 1 < dot(beta_10[1:], beta_10[1:]) < 2
    assert 0.5 < multiple_r_squared(inputs, daily_minutes_good, beta_10) < 0.6
    
if __name__ == "__main__": main()




