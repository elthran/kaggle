import numpy as np


class MathHelper:
    def __init__(self):
        pass

    @staticmethod
    def modified_powerlaw_function(x, a, b, c):
        """a modified form of a power law function: https://en.wikipedia.org/wiki/Power_law#Power-law_functions"""
        return c + x ** a * b

    @staticmethod
    def heavily_modified_logarithmic_function(x, a, b, c):
        """a heavily modified form of the logarithmic funtion: https://en.wikipedia.org/wiki/Logarithm#Logarithmic_function"""
        return a * np.log2(b + x) + c

    @staticmethod
    def generalized_logistic_function(x, a, b, c, d, g):
        """a modified form of the generalized logistic function: https://en.wikipedia.org/wiki/Generalised_logistic_function
        a the lower asymptote
        b the Hill coefficient, i.e. the steepness of the slope in the linear portion of the sigmoid
        c is related to the value Y(0), and is the inflection point of the curve, i.e. the x value of the middle of the the linear portion of the curve
        d the upper asymptote
        g asymmetry factor - set to 0.5 initially"""
        return ((a - d) / ((1 + ((x / c) ** b)) ** g)) + d
