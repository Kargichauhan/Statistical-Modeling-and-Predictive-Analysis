# Instructor: Professor Adarsh
# Description: This assignment contains a script   
# Collabrators: Rachel, Nees and Xiuchen Lu

# ISTA 421 / INFO 521 Fall 2023, HW 3, Exercises 5 and 6
# Author: Clayton T. Morrison
# This file is only made available for use in your submission for Homework 3
# of the current year (2023).
# You are NOT permitted to share this file with other students outside of
# this course year. You are also not permitting to post this file online
# except within your github classroom repository for this assignment.
# Doing any of those things is considered cheating by you and others not
# in the current course year. Cheating can be assessed retroactively, even
# after you graduate.

# Extension of predictive_variance_example.py
# Port of predictive_variance_example.m
# From A First Course in Machine Learning, Chapter 2.
# Simon Rogers, 01/11/11 [simon.rogers@glasgow.ac.uk]
# Predictive variance example

# References #
# FML Hardcopy 
# Section 02 Lectures starting 

import numpy
import os
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# Control exercise execution
# --------------------------------------------------------------------------

# Set each of the following lines to True in order to make the script execute
# each of the exercise functions

RUN_EXERCISE_5A = True
RUN_EXERCISE_5B = True
RUN_EXERCISE_5C = True
RUN_EXERCISE_6 = True


# --------------------------------------------------------------------------
# Globals
# --------------------------------------------------------------------------

DATA_ROOT = os.path.join('..', 'data')
PATH_TO_SYNTH_DATA = os.path.join(DATA_ROOT, 'synth_data.csv')
PATH_TO_SYNTH_DATA_SETS = os.path.join(DATA_ROOT, 'synth_data_sets.csv')

FIGURES_ROOT = os.path.join('..', 'figures')
PATH_TO_SYNTH_DATA_FIG = os.path.join(FIGURES_ROOT, 'ex5_synth_data.png')
PATH_TO_EX5B_FN_NAME_BASE = os.path.join(FIGURES_ROOT, 'ex5b_fn_errorbars_order')
PATH_TO_EX5C_FN_NAME_BASE = os.path.join(FIGURES_ROOT, 'ex5c_sample_fn_order')
PATH_TO_EX6_FN_NAME_BASE = os.path.join(FIGURES_ROOT, 'ex6_sample_fn_order')


# --------------------------------------------------------------------------
# You must complete the implementation of the following two functions
# --------------------------------------------------------------------------

def calculate_prediction_variance(x_new, X, w, t):
    """
    Calculates the variance for the prediction at x_new
    :param X: Design matrix: matrix of N observations
    :param w: vector of parameters
    :param t: vector of N target responses
    :param x_new: new observation
    :return: the predictive variance around x_new
    """
    ### Insert your code here ###
    predictive_variance = 0  # NOTE: 0 is NOT the solution!
    N = X.shape[0] #observation 
    D = X.shape[1] #num of columns
    #biased = sigma_2(1-D/N) therefore to remove it divide by (1-D/N)
    

    sigma_2 = ((1/(N))*(numpy.dot(t.T,t)-numpy.dot((numpy.dot(t,X)),w)))/(1-D/N)
    predictive_variance = (numpy.dot(numpy.dot(sigma_2 * x_new.T, numpy.linalg.inv(numpy.dot(X.T, X))), x_new))
    #Calculate the predictive variance
    #predictive_variance = 0  # NOTE: 0 is NOT the solution!
    #predictive_variance = sigma_2 * numpy.diag(x_new*numpy.linalg.inv(X.T*X)*testX.T)
    #print (predictive_variance)
    #print(N)
    #print(D)
    

    
    return predictive_variance


def calculate_cov_w(X, w, t):
    """
    Calculates the covariance of w
    :param X: Design matrix: matrix of N observations
    :param w: vector of parameters
    :param t: vector of N target responses
    :return: the matrix covariance of w
    """
    ### Insert your code here ###
    N = X.shape[0] # observations
    D = X.shape[1] # num of columns
    # Calculate the estimated covariance of w
    sigma_2 = ((1.0/(N))*(numpy.dot(t.T,t)-numpy.dot((numpy.dot(t,X)),w)))/(1-D/N)
    covw = numpy.eye(w.shape[0])  # NOTE: numpy.eye(w.shape[0]) is NOT the solution!
    covw = sigma_2*numpy.linalg.inv(numpy.dot(X.T, X))
    #covw = numpy.linalg.inv(numpy.matmul(numpy.transpose(X),X))
    

    return covw


# --------------------------------------------------------------------------
# Part 5a -- Generate and Plot the data
# There is nothing you need to implement here
# --------------------------------------------------------------------------

def true_function(x):  # 2023
    """$t = 5-2x-3x^2+0.5x^3$"""
    return 5 - (2 * x) - (3 * x**2) + (0.5 * x**3)


def read_data_from_file(filepath):
    data = numpy.loadtxt(filepath, delimiter=',')
    x = data[:, 0]
    t = data[:, 1]
    return x, t


def save_data_to_file(x, t, filepath):
    data = numpy.stack((x, t), axis=1)  # x first column, y second column
    numpy.savetxt(filepath, data, delimiter=',')


def sample_from_function(N=100, noise_var=1000, xmin=-5., xmax=5.):
    """
    Sample data from the true function.
    :param N: Number of samples
    :param noise_var: noise variance
    :param xmin: plot x min
    :param xmax: plot x max
    :return: x, t (arrays)
    """
    x = numpy.random.uniform(xmin, xmax, N)
    t = true_function(x)
    # add standard normal noise using numpy.random.randn
    # (standard normal is a Gaussian N(0, 1.0)  (i.e., mean 0, variance 1),
    #  so multiplying by numpy.sqrt(noise_var) make it N(0,standard_deviation))
    t = t + numpy.random.randn(x.shape[0])*numpy.sqrt(noise_var)
    return x, t


def plot_data(x, t, xmin, xmax, synth_data_figure_path=None):

    # Plot just the sampled data
    plt.figure(0)
    plt.scatter(numpy.asarray(x), numpy.asarray(t), color='k', edgecolor='k')

    # find ylim plot bounds based on data
    envelope = (max(t) - min(t)) * 0.1
    ymin = min(t) - envelope
    ymax = max(t) + envelope
    plt.ylim(ymin, ymax)

    plt.xlabel('x')
    plt.ylabel('t')
    plt.title(r'Sampled data from {0}, $x \in [{1},{2}]$'
              .format(true_function.__doc__, xmin, xmax))
    plt.pause(.1)  # required on some systems so that rendering can happen

    if synth_data_figure_path is not None:
        plt.savefig(synth_data_figure_path, format='png')


# --------------------------------------------------------------------------
# Part 5b -- Error bar plots
# There is nothing you need to implement here;
# However, you need to complete the implementation of function
# calculate_prediction_variance, above
# --------------------------------------------------------------------------

def plot_with_error_bars(x, t, orders, xmin, xmax, fn_errorbars_figure_path_base):
    """
    Generate plots of predicted variance (error bars) for various model orders
    :param orders:
    :param xmin:
    :param xmax:
    :return:
    """

    # Make a set of 100 evenly-spaced x values between xmin and xmax
    testx = numpy.linspace(xmin, xmax, 100)

    for i in orders:
        # create input representation for given model polynomial order
        X = numpy.zeros(shape=(x.shape[0], i + 1))
        testX = numpy.zeros(shape=(testx.shape[0], i + 1))
        for k in range(i + 1):
            X[:, k] = numpy.power(x, k)
            testX[:, k] = numpy.power(testx, k)

        # fit model parameters
        w = numpy.dot(numpy.linalg.inv(numpy.dot(X.T, X)), numpy.dot(X.T, t))

        # calculate predictions
        prediction_t = numpy.dot(testX, w)

        # get variance, each x_new at a time
        prediction_t_variance = numpy.zeros(testX.shape[0])
        for j in range(testX.shape[0]):
            x_new = testX[j, :]  # get the x_new observation from row i of testX
            var = calculate_prediction_variance(x_new, X, w, t)  # calculate prediction variance
            prediction_t_variance[j] = var

        # Plot the data and predictions
        plt.figure()
        plt.scatter(x, t, color='k', edgecolor='k')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.errorbar(testx, prediction_t, prediction_t_variance)


        # ylim bounds based on predicted t with variance!!!!!!!!!!
        # min_model = min(prediction_t - prediction_t_variance)
        # max_model = max(prediction_t + prediction_t_variance)
        # min_testvar = min(min(t), min_model)
        # max_testvar = max(max(t), max_model)
        # plt.ylim(min_testvar, max_testvar)


        # find ylim plot bounds based on data
        envelope = (max(t) - min(t)) * 0.1
        ymin = min(t) - envelope
        ymax = max(t) + envelope
        plt.ylim(ymin, ymax)

        ti = 'Plot of predicted variance for model with polynomial order {:g}'.format(i)
        plt.title(ti)
        plt.pause(.1)  # required on some systems so that rendering can happen

        # Save the figure
        full_figure_path = f'{fn_errorbars_figure_path_base}-{i}.png'
        plt.savefig(full_figure_path, format='png')


# --------------------------------------------------------------------------
# Part 5c -- Plot functions by sampling from cov(\hat{w})
# There is nothing you need to implement here.
# However, you need to complete the implementation of function
# calculate_cov_w, above
# --------------------------------------------------------------------------

def plot_functions_sampling_from_covw(x, t, orders, num_function_samples,
                                      xmin, xmax, sampled_fn_figure_path_base):

    # Make a set of 100 evenly-spaced x values between xmin and xmax
    testx = numpy.linspace(xmin, xmax, 100)

    for i in orders:
        # create input representation for given model polynomial order
        X = numpy.zeros(shape=(x.shape[0], i+1))
        testX = numpy.zeros(shape=(testx.shape[0], i+1))

        for k in range(i + 1):
            X[:, k] = numpy.power(x, k)
            testX[:, k] = numpy.power(testx, k)

        # fit model parameters
        w = numpy.dot(numpy.linalg.inv(numpy.dot(X.T, X)), numpy.dot(X.T, t))

        # Sample functions with parameters w sampled from a Gaussian with
        # $\mu = \hat{\mathbf{w}}$
        # $\Sigma = cov(w)$

        # determine cov(w)
        covw = calculate_cov_w(X, w, t)  # calculate the covariance of w

        # The following samples num_function_samples of w from
        # multivariate Gaussian (normal) with covaraince covw
        wsamp = numpy.random.multivariate_normal(w, covw, num_function_samples)

        # Calculate means for each function
        prediction_t = numpy.dot(testX, wsamp.T)

        # Plot the data and functions
        plt.figure()
        plt.scatter(x, t, color='k', edgecolor='k')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.plot(testx, prediction_t, color='b')


        # find ylim plot bounds based on data
        envelope = (max(t) - min(t)) * 0.1
        ymin = min(t) - envelope
        ymax = max(t) + envelope
        plt.ylim(ymin, ymax)

        ti = 'Plot of {0} functions where parameters '\
             .format(num_function_samples, i) + \
             r'$\widehat{\bf w}$ were sampled from' + '\n' + r'cov($\bf w$)' + \
             ' of model with polynomial order {1}' \
             .format(num_function_samples, i)
        plt.title(ti)
        plt.pause(.1)  # required on some systems so that rendering can happen

        # Save the figure
        full_figure_path = f'{sampled_fn_figure_path_base}-{i}.png'
        plt.savefig(full_figure_path, format='png')


# --------------------------------------------------------------------------
# Exercise 6
# You do need to fill in the implementation of the inner loop in exercise_6
# --------------------------------------------------------------------------

def plot_ex6_figure_setup(num_sample_sets, sample_size, model_order):
    plt.xlabel('x')
    plt.ylabel('t')
    plt.ylim(-40, 40)  # (-800,800)
    plt.title(f'Plot of {num_sample_sets} functions, '
              f'each a best fit polynomial order {model_order} model\n'
              f'to one of {num_sample_sets} sampled data sets of size {sample_size}.')


def exercise_6(orders, noise_var, xmin, xmax,
               data_sets_source_path, sampled_fn_figure_path_base):
    # Make a set of 100 evenly-spaced x values between xmin and xmax
    testx = numpy.linspace(xmin, xmax, 100)

    # Generate common set of data
    num_sample_sets = 20
    sample_size = 25
    sample_sets = list()


    def read_data():
        data = numpy.loadtxt(data_sets_source_path, delimiter=',')
        for i in range(num_sample_sets):
            data_set = data[i*sample_size:(i+1)*sample_size, :]
            x, t = data_set[:, 0], data_set[:, 1]
            sample_sets.append( (x, t) )

    read_data()

    # for checking solution
    ex6_results = list()

    # For each polynomial order, find the best-fit model and plot it for each data set
    # Also plot the mean of the true generating function
    for model_order in orders:

        # create a new figure
        plt.figure()
        plot_ex6_figure_setup(num_sample_sets, sample_size, model_order)

        # For checking solution
        last_w = None
        last_t = None

        # For each sample set, find the best fit model of order i and add its plot (in blue)
        for (x, t) in sample_sets:

            ### Insert your code here ###
            # create input representation for given model polynomial order
            X = numpy.zeros(shape=(x.shape[0], model_order + 1))
            testX = numpy.zeros(shape=(testx.shape[0], model_order + 1))

            for k in range(model_order + 1):
                X[:, k] = numpy.power(x, k)
                testX[:, k] = numpy.power(testx, k)

            # fit model mean parameters
            #w = numpy.zeros(model_order + 1)  # numpy.zeros(model_order + 1) is not the right answer!
            w = numpy.dot(numpy.linalg.inv(numpy.dot(X.T,X)), numpy.dot(X.T, t))
            # calculate predicted mean
            prediction_t = numpy.dot(testX, w) # numpy.zeros(x.shape[0]) is not the right answer!
            
            if prediction_t is not None:
                # plot the best-fit model to the current sample, in blue
                plt.plot(testx, prediction_t, color='b')
                # update solution
                last_t = prediction_t[0]
                last_w = w

        # plot the true model, in red, on top of previously function plots
        plt.plot(testx, true_function(testx), linewidth=3, color='r')
        plt.pause(.1)  # required on some systems so that rendering can happen

        ex6_results.append((last_w, last_t))

        # Save the figure
        full_figure_path = f'{sampled_fn_figure_path_base}-{model_order}.png'
        plt.savefig(full_figure_path, format='png')

    return tuple(ex6_results)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # parameters of the synthetic data
    xmin = -2.
    xmax = 7.
    noise_var = 6


    # If you uncomment the next line (calling sample_from_function) and
    # comment-out the line after (calling read_data_from_file), then
    # you can sample 100 new points from the function. This is worth doing
    # to see how the error bars behave under different samples.
    # However, for your submission, you should load the pre-generated
    # data from file PATH_TO_SYNTH_DATA; this is what the unit tests
    # are expecting; so if you do try sample_from_function, be sure to
    # revert back to having that commented-out and NOT commenting out the
    # call to read_data_from_file(PATH_TO_SYNTH_DATA).
    #
    # x, t = sample_from_function(40, noise_var, xmin, xmax)
    # Read synthetic data from file
    x, t = read_data_from_file(PATH_TO_SYNTH_DATA)

    # the min and max of the interval of data points being removed (for parts 5a and 5c)
    xmin_remove = 2.5
    xmax_remove = 4.5

    # Chop out some x data:
    # the following line expresses a boolean function over the values in x;
    # this produces a list of the indices of list x for which the test
    # was not met; these indices are then deleted from x and t.
    pos = ((x >= xmin_remove) & (x <= xmax_remove)).nonzero()

    x = numpy.delete(x, pos, 0)
    t = numpy.delete(t, pos, 0)

    # Fit models of various orders
    orders = (1, 3, 5, 9)

    # for 5c
    # Generate plots of functions whose parameters are sampled based on cov(\hat{w})
    num_function_samples = 20

    if RUN_EXERCISE_5A:
        plot_data(x, t, xmin, xmax, PATH_TO_SYNTH_DATA_FIG)
    if RUN_EXERCISE_5B:
        plot_with_error_bars(x, t, orders, xmin, xmax, PATH_TO_EX5B_FN_NAME_BASE)
    if RUN_EXERCISE_5C:
        plot_functions_sampling_from_covw(x, t, orders, num_function_samples,
                                          xmin, xmax, PATH_TO_EX5C_FN_NAME_BASE)
    if RUN_EXERCISE_6:
        exercise_6(orders, noise_var, xmin, xmax,
                   PATH_TO_SYNTH_DATA_SETS, PATH_TO_EX6_FN_NAME_BASE)
    plt.show()
