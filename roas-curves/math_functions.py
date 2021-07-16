import numpy as np


class MathHelper:
    def __init__(self):
        self.all_functions = [
            {"name": "modified_powerlaw_function", "function": self.modified_powerlaw_function},
            {"name": "heavily_modified_logarithmic_function", "function": self.heavily_modified_logarithmic_function},
            {"name": "generalized_logistic_function", "function": self.generalized_logistic_function},
        ]

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
        # return (np.abs((a - d) / ((1 + ((x / c) ** np.abs(b))) ** np.abs(g))) + d
        return ((a - d) / ((1 + ((x / c) ** b)) ** g)) + d

    def get_functions(self, function_name):
        """Returns a list of all functions or a list with only the chosen function if passed in"""
        if function_name:
            return [function_dict for function_dict in self.all_functions if function_dict["name"] == function_name]
        else:
            return self.all_functions
