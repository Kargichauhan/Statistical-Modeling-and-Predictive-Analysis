# Instructor: Professor Adarsh
# Description: This assignment includes a demonstration of how to approximate an expected value through sampling. 
# Collabrators: Rachel, Nees and Xiuchen Lu

## Note: We got extension from professor till 11.59pm to submit because of the exam on the next day ##

# ISTA 421 / INFO 521 Fall 2023, HW 3, Exercise 2
# Author: Clayton T. Morrison
# This file is only made available for use in your submission for Homework 3
# of the current year (2023).
# You are NOT permitted to share this file with other students outside of
# this course year. You are also not permitting to post this file online
# except within your github classroom repository for this assignment.
# Doing any of those things is considered cheating by you and others not
# in the current course year. Cheating can be assessed retroactively, even
# after you graduate.

# approx_expected_value_sin.py
# Port of approx_expected_value.m
# From A First Course in Machine Learning, Chapter 2.
# Simon Rogers, 01/11/11 [simon.rogers@glasgow.ac.uk]
# Approximating expected values via sampling

# References #
# FML Hardcopy 
# Section 02 Lectures starting 
import numpy
import os
import matplotlib.pyplot as plt
import random_generator


# -----------------------------------------------------------------------------
# Control exercise execution
# -----------------------------------------------------------------------------

# Set each of the following lines to True in order to make the script execute
# each of the exercise functions

RUN_DEMO = True
RUN_EXERCISE_2 = True


# -----------------------------------------------------------------------------
# Global paths
# -----------------------------------------------------------------------------

DATA_ROOT = os.path.join('..', 'data')
PATH_TO_RANDOM_UNIFORM_10000 = os.path.join(DATA_ROOT, 'rand_uniform_10000.txt')

FIGURES_ROOT = os.path.join('..', 'figures')
PATH_TO_FN_APPROX_FIG = os.path.join(FIGURES_ROOT, 'ex2_fn_approx.png')


# -----------------------------------------------------------------------------
# Sampling demo code
# -----------------------------------------------------------------------------

# We are trying to estimate the expected value of
# $f(y) = y^2$
##
# ... where
# $p(y)=U(0,1)$
##
# ... which is given by ('\int' is the latex code for "integral"):
# $\int y^2 p(y) dy$
##
# The exact result is:
# $\frac{1}{3}$ = 0.333...
# (NOTE: this just gives you the result -- you should be able to derive it!)

# First, let's plot the function, shading the area under the curve for x=[0,1]
# The following plot_fn helps us do this.

# Information about font to use when plotting the function
FONT = {'family' : 'serif',
        'color'  : 'darkred',
        'weight' : 'normal',
        'size'   : 12,
        }


def plot_fn(fn, a, b, fn_name=None, resolution=100):
    """
    Plots a function fn between lower bound a and upper bound b.
    :param fn: a function of one variable
    :param a: lower bound
    :param b: upper bound
    :param fn_name: the name of the function (displayed in the plot)
    :param resolution: the number of points between a and b used to plot the fn curve
    :return: None
    """
    x = numpy.append(numpy.array([a]),
                     numpy.append(numpy.linspace(a, b, resolution), [b]))
    y = fn(x)
    y[0] = 0
    y[-1] = 0
    plt.figure()
    plt.fill(x, y, 'b', alpha=0.3)
    if fn_name:
        fname, x_tpos, y_tpos = fn_name()
        plt.text(x_tpos, y_tpos, fname, fontdict=FONT)
    plt.title('Area under function')
    #plt.xlabel('$y$')
    #plt.ylabel('$f(y)$')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')


    x_range = b - a

    plt.xlim(a-(x_range*0.1), b+(x_range*0.1))


# Define the function y that we are going to plot
def y_fn(x):
    """
    The function x^2
    :param x:
    :return:
    """
    return numpy.power(x, 2)
    



def y_fn_name():
    """
    Helper for displaying the name of the fn in the plot
    Returns the parameters for plotting the y function name,
    used by approx_expected_value
    :return: fname, x_tpos, y_tpos
    """
    #fname = r'$f(y) = y^2$'  # latex format of fn name
    fname = r'$f(x) = 35+3x-3x^2+0.2x^3+0.01x^4$' # latex format of fn
    x_tpos = 0.1  # x position for plotting text of name
    y_tpos = 0.5  # y position for plotting text of name
    return fname, x_tpos, y_tpos


def run_sample_demo():
    # Plot the function!
    plot_fn(y_fn, 0, 1, y_fn_name)

    # Now we'll approximate the area under the curve using sampling...

    # Sample 1000 uniformly random values in [0..1]
    ys = numpy.random.uniform(low=0.0, high=1.0, size=1000)
    # compute the expectation of y, where y is the function that squares its input
    ey2 = numpy.mean(numpy.power(ys, 2))
    print('\nSample-based approximation: {:f}'.format(ey2))

    # Store the evolution of the approximation, every 10 samples
    sample_sizes = numpy.arange(1, ys.shape[0], 10)
    ey2_evol = numpy.zeros((sample_sizes.shape[0]))  # storage for the evolving estimate...
    # the following computes the mean of the sequence up to i, as i iterates
    # through the sequence, storing the mean in ey2_evol:
    for i in range(sample_sizes.shape[0]):
        ey2_evol[i] = numpy.mean(numpy.power(ys[0:sample_sizes[i]], 2))

    # Create plot of evolution of the approximation
    plt.figure()
    # plot the curve of the estimation of the expected value of f(x)=y^2
    plt.plot(sample_sizes, ey2_evol)
    # The true, analytic result of the expected value of f(y)=y^2 where y ~ U(0,1): $\frac{1}{3}$
    # plot the analytic expected result as a red line:
    plt.plot(numpy.array([sample_sizes[0], sample_sizes[-1]]),
             numpy.array([1./3, 1./3]), color='r')
    plt.xlabel('Sample size')
    plt.ylabel('Approximation of expectation')
    plt.title('Approximation of expectation of $f(y) = y^2$')
    plt.pause(.1)  # required on some systems so that rendering can happen


# -----------------------------------------------------------------------------
# Exercise 2
# -----------------------------------------------------------------------------

def exercise_2(path_to_random_numbers, figure_path):
    """
    Provide code to
    :param path_to_random_numbers: directory path to random number source file
    :return: expected, estimate
    """

    # Do not edit the following.
    # Use the following random number generator (rng) to draw random samples
    # This will allow you to generate Uniform random samples (this will load
    # the random numbers from file, and they're already set up to be in the
    # uniform random range for this assignment)
    rng = random_generator.RNG(path_to_random_numbers)

    # In your code, rather than calling
    #     numpy.random.uniform(low, high, size)
    # instead use the following function to draw n (i.e., size) samples.
    # Like numpy.random.uniform, this will return an array of uniform random
    # numbers, of length n:
    #     rng.get_n_random(n)

    #### YOUR CODE HERE ####
    sample_data = rng.get_n_random(10000)
    ey2_fill = numpy.zeros(len(sample_data))
    for i in range(len(ey2_fill)):
        calc = 35 + (3* sample_data[i]) - (3 * sample_data[i] **2) + (.2*sample_data[i]**3)+ (.01*sample_data[i]**4)
        ey2_fill[i] = calc
    ey2 = numpy.mean(ey2_fill)
    print('\nSample-based approximation: {:f}'.format(ey2))
    # Store the evolution of the approximation, every 10 samples
    sample_sizes = numpy.arange(1, sample_data.shape[0], 10)
    ey2_evol = numpy.zeros((sample_sizes.shape[0]))  # storage for the evolving estimate...
    # the following computes the mean of the sequence up to i, as i iterates
    # through the sequence, storing the mean in ey2_evol:
    for i in range(sample_sizes.shape[0]):
        x = sample_data[0:sample_sizes[i]]
        calc = 35 + (3*x) - (3*x**2) + (.2*x**3)+ (.01*x**4)
        ey2_evol[i] = numpy.mean(calc)
   
    #Calculating the integral of the equation 
    #35+3x-3x^2 + .2x^3 + .01x^4 times uniform distribution
    #35+3x-3x^2 + .2x^3 + .01x^4 times 1/ b-a
    #find indefinate integral of equation plug in 9 and subtract -1
    #35x + 3/2 x^2 - 3/3 x^3 + 0.2/4 x ^ 4 + .01/5 x^ 5
    #Plug in 9 and subtract -1
    #     ey2 = numpy.mean(35+3*ys-0.5*numpy.power(ys, 3)+0.05*numpy.power(ys, 4))

    integralb = (35 * 9) + (3/2 * 9 ** 2) - (1 * 9 ** 3) + (.2/4 * 9 ** 4) + (.01/5 * 9 ** 5)
    b1 = 35 * -1
    b2 = 1.5*(-1)**2
    b3 = 1*(-1)**3
    b4 = .05 *(-1) **4
    b5 = .002 *(-1) **5
    integral1 = b1 + b2 - b3 + b4 + b5


    #needed to round because of the recurring zeros 

    ey2_exp = (round(integralb, 3) - round(integral1, 3))/ 10


    # Create plot of evolution of the approximation
    plt.figure()
    # plot the curve of the estimation of the expected value of f(x)=y^2
    plt.plot(sample_sizes, ey2_evol)
    # The true, analytic result of the expected value of f(y)=y^2 where y ~ U(0,1): $\frac{1}{3}$
    # plot the analytic expected result as a red line:
    plt.plot(numpy.array([sample_sizes[0], sample_sizes[-1]]), numpy.array([ey2_exp, ey2_exp]), color='r')
    plt.xlabel('Sample size')
    plt.ylabel('Approximation of expectation')
    plt.title('Approximation of expectation of $f(x) = 35 + 3x - 3x^2 + 0.2x^3 + 0.01x^4$')
    plt.pause(.1)  # required on some systems so that rendering can happen

    if figure_path:
        plt.savefig(figure_path, format='png')

    expected = ey2_exp # Calculate the expected value; 
    estimate = ey2  # Calculate the sample approximation of the expected value; 

    print(f'Expectation: {expected}')
    print(f'Sample-based approximation: {estimate}')

    return expected, estimate



# -----------------------------------------------------------------------------
# TOP LEVEL SCRIPT
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    if RUN_DEMO:
        run_sample_demo()
        plt.show()
    if RUN_EXERCISE_2:
        exercise_2(PATH_TO_RANDOM_UNIFORM_10000, PATH_TO_FN_APPROX_FIG)
        plt.show()
