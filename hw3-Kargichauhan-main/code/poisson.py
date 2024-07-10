# Instructor: Professor Adarsh
# Description: This assignment contains a script   
# Collabrators: Rachel, Nees and Xiuchen Lu

# ISTA 421 / INFO 521 Fall 2023, HW 3, Exercise 1
# Author: Clayton T. Morrison
# This file is only made available for use in your submission for Homework 3
# of the current year (2023).
# You are NOT permitted to share this file with other students outside of
# this course year. You are also not permitting to post this file online
# except within your github classroom repository for this assignment.
# Doing any of those things is considered cheating by you and others not
# in the current course year. Cheating can be assessed retroactively, even
# after you graduate.

# References #
# FML Hardcopy 
# Section 02 Lectures starting 


import math
# -----------------------------------------------------------------------------
# Control exercise execution
# -----------------------------------------------------------------------------

# Set each of the following lines to True in order to make the script execute
# each of the exercise functions


RUN_EX1_A = True
RUN_EX1_B = True


# -----------------------------------------------------------------------------
# Exercises
# -----------------------------------------------------------------------------


def calculate_poisson_pmf_a():
    """
    Calculate probability that Y ~ Poisson(lambda=5) for 3 <= Y <= 7
    :return: probability
    """
    ### YOUR CODE HERE
    probability = 0  # NOTE: 0 is not the correct answer!
    Yminor = 2
    Ymax = 6
    lammda = 3

    #loop to sum all probability from 3 to 7
    for x in range(Yminor,Ymax+1):
        probability += (lammda**x/math.factorial(x))*math.exp(-lammda)
        #print(probability)

    #print(probability)
    return probability


def calculate_poisson_pmf_b():
    """
    Calculate probability that Y ~ Poisson(lambda=5) for Y < 3 or Y > 7
    :return: probability
    """
    ### YOUR CODE HERE
    probability = 0  #  NOTE: 0 is not the correct answer!
    #probability of y < 3 or y < 7 is 1- (y>=3 and y <=7)
    probability = 1-calculate_poisson_pmf_a()
    #print(probability)
    return probability


# -----------------------------------------------------------------------------
# TOP LEVEL SCRIPT
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    if RUN_EX1_A:
        calculate_poisson_pmf_a()
    if RUN_EX1_B:
        calculate_poisson_pmf_b()
