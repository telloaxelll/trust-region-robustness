import numpy as np
import matplotlib.pyplot as plt

# This file contains utility functions for generating synthetic data based on specific 
# mathematical models used to test optimization algorithms. The functions defined here are sourced 
# from the following references: 
# 1. Rosenbrock Function: https://en.wikipedia.org/wiki/Rosenbrock_function
# 2. Beale Function: https://en.wikipedia.org/wiki/Beale_function

def rosenbrock(a=1, b=100):
    """
    Case 1: Nonlinear & Non-convex Function

    This function defines model that is being used to generate our synthetic data. 
    We generate the data based on the 'Rosenbrock Function` which serves as general test case for 
    optimization algorithms. 

    Args:
        a (int): The a parameter for the Rosenbrock function.
        b (int): The b parameter for the Rosenbrock function.

    Returns: 
        array: The function value at the given input.
    """

    x = np.linspace(0, 100, 1000)
    y = np.linspace(0, 100, 1000)

    return (a - x) ** 2 + b * (y - x ** 2) ** 2

def beale():
    """
    Case 2: Ill-Conditioned Function

    This function defines a model that is being used to generate our synthetic data.
    We generate the data based on the 'Beale Function' which serves as a test case for
    optimization algorithms under ill-conditioning, where the Hessian condition number
    grows large away from the solution, stressing step-size control mechanisms.

    Args:
        None

    Returns:
        array: The function value at the given input.
    """
    x = np.linspace(-4.5, 4.5, 1000)
    y = np.linspace(-4.5, 4.5, 1000)
    
    return (1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2