import random
import copy
import time
import math
import numpy as np
from sys import argv
from datetime import datetime
from math import sqrt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error


class SVM(object):

    def __init__(self, c, gamma):
        self.svm = SVR(C=c, gamma=gamma)
        self.c = c
        self.gamma = gamma
        self.fit = -1
        self.chance = -1


    def __str__(self):
        """Print the individual"""
        return str(self.c) + " ; " + str(self.gamma) + " ; " + str(self.fit)

    def calculate_fitness(self, data):
        """Calculate the fitness of the solution"""
        if self.fit == -1:
            self.svm.fit(data[0], data[1])
            predicted = self.svm.predict(data[2])
            self.fit = sqrt(mean_squared_error(data[3], predicted))

    def set_params(self):
        self.svm.set_params(C=self.c, gamma=self.gamma)

class GA(object):

    def __init__(self, mt_rate, cx_rate, elt_rate, pop_size, max_gen, data):
        self.mt_rate = mt_rate
        self.cx_rate = cx_rate
        self.elt_rate = elt_rate
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.pop = []
        self.data = data

    def random_ind(self):
        """Create a random individual"""
        c = random.uniform(0, 1)
        gamma = random.uniform(0, 1)
        return SVM(c,gamma)

    def start_population(self):
        """Start the population with random individuals"""
        for i in range(self.pop_size):
            ind = self.random_ind()
            self.pop.append(ind)

    def mutation(self,ind):
        """Mutation operator."""
        if probabilty(self.mt_rate):
            k = random.randint(0,1)
            r = random.uniform(0,1)
            if (k == 0):
                ind.c = ind.c*r
            else:
                ind.gamma = ind.gamma*r
            ind.set_params()

    def crossover(self,ind1,ind2):
        """crossover operator. Trade lines between individuals"""
        new_ind1 = copy.deepcopy(ind1)
        new_ind1.fit = -1
        new_ind2 = copy.deepcopy(ind2)
        new_ind2.fit = -1
        if probabilty(self.cx_rate):
            aux = new_ind1.gamma
            new_ind1.gamma = new_ind2.gamma
            new_ind2.gamma = aux
            new_ind1.set_params()
        return new_ind1, new_ind2

    def roulette_selection(self):
        """Fitness proportionate random selection."""
        tot = sum([ind.chance for ind in self.pop])
        r = random.uniform(0,tot)
        for ind in self.pop:
            r -= ind.chance
            if r <= 0:
                ind.chance = 0
                return ind

    def tournament_selection(self):
        """Tournament selection with size 2"""
        r1 = random.randint(0,len(self.pop)-1)
        r2 = random.randint(0,len(self.pop)-1)
        if self.pop[r1].fit < self.pop[r2].fit:
            return self.pop[r1]
        return self.pop[r1]

    def calculate_population_fitness(self):
        """Calculate the fitness of each individual in population"""
        for ind in self.pop:
            ind.calculate_fitness(self.data)

    def start(self):
        """Start the evolution process"""
        self.start_population()
        self.calculate_population_fitness()
        gen = 1
        while gen < self.max_gen:
            new_population = []
            self.pop.sort(key = lambda ind: ind.fit)
            print "Generation: ", gen, " Fittest: ", self.pop[0]
            while len(new_population) < self.pop_size:
                for ind in self.pop:
                    ind.chance = float(1)/ind.fit
                ind1 = self.roulette_selection()
                ind2 = self.roulette_selection()
                new_ind = self.crossover(ind1, ind2)
                self.mutation(new_ind[0])
                self.mutation(new_ind[1])
                new_ind[0].calculate_fitness(self.data)
                new_population.append(new_ind[0])
                new_ind[1].calculate_fitness(self.data)
                new_population.append(new_ind[1])
            new_population.sort(key = lambda ind: ind.fit)
            new_population = new_population[0:int((1-self.elt_rate)*self.pop_size)]
            self.pop = self.pop[0:int(self.elt_rate*self.pop_size)]
            self.pop = self.pop + new_population
            gen +=1

def probabilty(prob):
    "Given a probabilty determines if the event is going to happen or not"
    r = random.random()
    return  r <= prob

def readCSV():
    "Read dataset from CSV file"
    data = np.loadtxt('data3.csv', dtype=float, delimiter=',', skiprows=1)
    np.random.shuffle(data)
    training_size = math.floor(data.shape[0]*0.9)
    training = data[:training_size,:]
    test = data[training_size:,:]
    training_x = training[:, 1:]
    training_y = training[:, 0]
    test_x = test[:, 1:]
    test_y = test[:, 0]

    return training_x, training_y, test_x, test_y

def main():
    random.seed(10)

    data = readCSV();
    mt_rate = 0.5
    cx_rate = 0.9
    elt_rate = 0.1
    pop_size = 100
    max_gen = 500

    ga = GA(mt_rate, cx_rate, elt_rate, pop_size, max_gen, data)
    ga.start()
    return 0

if __name__ == "__main__":
    main()
