'''
Author: Elly Mandliel
'''
import sys
import numpy as np
from EvoNN import *

class EvoNN(object):

    def __init__(self, input_size, shape, middle_activation, final_activation, epsilon):
        assert type(input_size) is int, "Input size should be integer"
        assert type(shape) in (tuple, list), "Shape should be tuple or list"
        assert callable(middle_activation), "Middle activation should be a function"
        assert callable(final_activation), "Final activation should be a function"
        assert 0 <= epsilon <= 1, "Epsilon should be between 0 and 1"

        self.shape = shape
        self.input_size = input_size
        self.middle_activation = middle_activation
        self.final_activation = final_activation
        self.epsilon = epsilon
        self.weights = []
        self.b = +0.1

    def normalized_random(self, *shape):
        return 2 * np.random.rand(*shape) - 1
    
    def initialize_variables(self):
        j = self.input_size
        for i in self.shape:
            self.weights.append(self.normalized_random(j, i))
            j = i

    def predict(self, x):
        for idx, weight in enumerate(self.weights):
            x = np.dot(x, weight) + self.b
            if idx == len(self.weights) - 1:
                x = self.middle_activation(x)
            else:
                x = self.final_activation(x)
        return x

    def mutate_weight(self, w0, w1):
        '''
        Mutate a single matrix.
        Getting +- from each parent and epsilon as mutate rate.
        '''
        new_weights = np.array(w0)
        random_booleans = np.random.rand() < 0.5
        new_weights[random_booleans] = w1[random_booleans]
        epsilon_randoms = np.random.rand() < self.epsilon
        new_weights[random_booleans] = 2 * np.random.rand() - 1
        return new_weights

    def set_weight(self, index, value):
        while len(self.weights) < index + 1:
            self.weights.append([])
        self.weights[index] = value

    def mutate(self, other):
        small_child = EvoNN(self.input_size,
                                        self.shape, 
                                        self.middle_activation, 
                                        self.final_activation, 
                                        self.epsilon)
        for index, value in enumerate(self.weights):
            small_child.set_weight(index, self.mutate_weight(other.weights[index], self.weights[index]))
        return small_child
