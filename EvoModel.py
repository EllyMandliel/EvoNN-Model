'''
Author: Elly Mandliel
'''
from EvoNN import *
import numpy as np

class EvoModel:
    '''
    A model of evolution of a neural network population.
    '''
    def __init__(self, model_args, volume, tester):
        '''
        model_args: EvoNN constructor arguments
        volume: maximum population size
        cycles: number of generations
        test: a function to evaluate an EvoNN's score.
        '''
        assert callable(tester), "Tester should be a function"
        try:
            EvoNN(**model_args)
        except Exception as e:
            e.args = ('Incorrect EvoNN arguments: ' + e.args[0],)
            raise 

        self.volume = volume
        self.tester = tester
        self.current_cycle = -1
        self.nns = [[EvoNN(**model_args), 0] for i in range(volume)]
        for nn in self.nns:
            nn[0].initialize_variables()

    def output_fitness(self):
        '''
        Returns the population score, max., min. and mean.
        '''
        scores = [i[1] for i in self.nns]
        scores.sort()
        scores_sum = sum(scores)
        scores_max = max(scores)
        scores_avg = scores_sum / len(scores)
        scores_median = scores[int(len(scores) / 2)]
        return 'Max: {}, Avg: {}, Median: {}'.format(
                scores_max,
                scores_avg,
                scores_median
            )

    def train(self, print_results = True):
        '''
        Cycle through generations in the model.
        '''
        self.current_cycle += 1
        for i in range(len(self.nns)):
            self.nns[i][1] = self.tester(self.nns[i][0])
        
        if print_results:
            scores = [i[1] for i in self.nns]
            scores.sort()
            scores_sum = sum(scores)
            scores_avg = scores_sum / len(scores)
            sc = scores[int(len(scores) / 2)]
            print('-' * 30)
            print('Cycle {}'.format(self.current_cycle))
            print(self.output_fitness())

    def eliminate(self, rate, print_fitness = True):
        '''
        Eliminate a fraction of the population.
        '''
        self.nns.sort(key=lambda x: x[1], reverse=True)
        eliminate_count = int(len(self.nns) * rate)
        for _ in range(eliminate_count):
            del self.nns[-1]
                

        if print_fitness:
            print('Total: {} Eliminated: {} Left: {}'.format(
                    len(self.nns) + eliminate_count,
                    eliminate_count,
                    len(self.nns)
                ))
            self.output_fitness()

    def repopulate(self, volume = None):
        '''
        Repopulate the network, selecting 2 random models and mutate them.
        '''
        if volume is not None:
            self.volume = volume
        
        while len(self.nns) < self.volume:
            # Repopulate 2 random models
            random_models = np.random.choice([i[0] for i in self.nns], size=2)
            self.nns.append([random_models[0].mutate(random_models[1]), 0])

def relu(x):
    x[x <= 0] = 0
    return x

def softmax(x):
    ex = np.exp(x)
    return ex / ex.sum(axis = 0)