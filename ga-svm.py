import random
import copy
import time
import math
import numpy as np
from sys import argv
from datetime import datetime
from math import sqrt
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


class SVM(object):

    def __init__(self, c, gamma, epsilon):
        self.svm = SVR(C=c, gamma=gamma)
        self.c = c
        self.gamma = gamma
        self.epsilon = epsilon
        self.fit = -1
        self.chance = -1


    def __str__(self):
        """Print the individual"""
        return str(self.c) + " ; " + str(self.gamma) + " ; " + str(self.epsilon) + " ; " + str(self.fit)

    def calculate_fitness(self, data):
        """Calculate the fitness of the solution"""
        if self.fit == -1:
            self.svm.set_params(C=self.c, gamma=self.gamma, epsilon=self.epsilon, tol=0.0002)
            self.svm.fit(data[0], data[1])
            predicted = self.svm.predict(data[2])
            self.fit = sqrt(mean_squared_error(data[3], predicted))

    def set_params(self):
        self.svm.set_params(C=self.c, gamma=self.gamma)

class PPA(object):

    def __init__(self, l_min, l_max, p_followup, tau, beta, pop_size, max_gen, data, params_lb, params_ub):
        self.l_min = l_min
        self.l_max = l_max
        self.p_followup = p_followup
        self.tau = tau
        self.beta = beta
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.prey = []
        self.pop = []
        self.data = data
        self.params_lb = params_lb
        self.params_ub = params_ub

    def random_ind(self):
        """Create a random individual"""
        c = random.uniform(self.params_lb[0], self.params_ub[0])
        gamma = random.uniform(self.params_lb[1], self.params_ub[1])
        epsilon = random.uniform(self.params_lb[2], self.params_ub[2])
        return SVM(c,gamma, epsilon)
    def start_population(self):
        """Start the population with random individuals"""
        for i in range(self.pop_size):
            ind = self.random_ind()
            self.pop.append(ind)

    def calculate_population_fitness(self):
        """Calculate the fitness of each individual in population"""
        for ind in self.pop:
            ind.calculate_fitness(self.data)

    def bounds(self, svm):
        if svm.c < self.params_lb[0]:
            svm.c = self.params_lb[0]
        if svm.c > self.params_ub[0]:
            svm.c = self.params_ub[0]
        if svm.gamma < self.params_lb[1]:
            svm.gamma = self.params_lb[1]
        if svm.gamma > self.params_ub[1]:
            svm.gamma = self.params_ub[1]
        if svm.epsilon < self.params_lb[2]:
            svm.epsilon = self.params_lb[2]
        if svm.epsilon > self.params_ub[2]:
            svm.epsilon = self.params_ub[2]

    def move_predator(self, predator):
        """Adjust values for the predator and add it to next generation"""
        x_prey = self.prey[np.argmax(x_j.fit for x_j in self.prey)]
        theta = random.random()*2*math.pi
        fi = random.random()*math.pi
        vector_c = x_prey.c - predator.c
        vector_gamma = x_prey.gamma - predator.gamma
        vector_epsilon = x_prey.epsilon - predator.epsilon
        dist = math.sqrt(vector_c*vector_c + vector_gamma*vector_gamma + vector_epsilon*vector_epsilon)
        sigma1 = random.random()
        sigma2 = random.random()
        predator.c += self.l_min*sigma1*math.sin(fi)*math.cos(theta)
        predator.gamma += self.l_min*sigma1*math.sin(fi)*math.sin(theta)
        predator.epsilon += self.l_min*sigma1*math.cos(fi)
        if dist > 0:
            predator.c += self.l_max*sigma2*vector_c/dist
            predator.gamma += self.l_max*sigma2*vector_c/dist
            predator.epsilon += self.l_max*sigma2*vector_epsilon/dist
        self.bounds(predator)
        self.pop.append(predator)

    def move_prey(self, x_i, predator):
        """Adjust values for one prey and add it to next generation"""
        A = []
        aux_c = 0
        aux_gamma = 0
        aux_epsilon = 0
        for x_j in self.prey:
            if x_j.fit < x_i.fit:
                A.append(x_j)
        if not A:
            pos = []
            k = 15
            for j in range(k):
                theta = random.random()*2*math.pi
                fi =  random.random()*math.pi
                sigma = random.random()
                s = SVM(x_i.c + self.l_min*sigma*math.sin(fi)*math.cos(theta),
                        x_i.gamma + self.l_min*sigma*math.sin(fi)*math.sin(theta),
                        x_i.epsilon + self.l_min*sigma*math.cos(fi) )
                if s.c <= self.params_ub[0] and s.c >= self.params_lb[0] and s.gamma <= self.params_ub[1] and s.gamma >= self.params_lb[1] and s.epsilon <= self.params_ub[2] and s.epsilon >= self.params_lb[2]:
                    pos.append(s)
            pos.append(x_i)
            for x in pos:
                x.calculate_fitness(self.data)
            x_i = pos[np.argmin(x_j.fit for x_j in pos)]
        else:
            if probabilty(self.p_followup):
                y_i_c = 0
                y_i_gamma = 0
                y_i_epsilon = 0
                for j in self.prey:
                    vector_c = j.c - x_i.c
                    vector_gamma = j.gamma - x_i.gamma
                    vector_epsilon = j.epsilon - x_i.epsilon
                    dist = math.sqrt(vector_c*vector_c + vector_gamma*vector_gamma + vector_epsilon*vector_epsilon)
                    y_i_c += math.exp(j.fit**self.tau - dist)*vector_c
                    y_i_gamma += math.exp(j.fit**self.tau - dist)*vector_gamma
                    y_i_epsilon += math.exp(j.fit**self.tau - dist)*vector_epsilon
                pos = []
                k = 15
                for j in range(k):
                    theta = random.random()*2*math.pi
                    fi =  random.random()*math.pi
                    s = SVM(x_i.c + self.l_min*math.sin(fi)*math.cos(theta),
                            x_i.gamma + self.l_min*math.sin(fi)*math.sin(theta),
                            x_i.epsilon + self.l_min*math.cos(fi) )
                    if s.c <= self.params_ub[0] and s.c >= self.params_lb[0] and s.gamma <= self.params_ub[1] and s.gamma >= self.params_lb[1] and s.epsilon <= self.params_ub[2] and s.epsilon >= self.params_lb[2]:
                        pos.append(s)
                pos.append(x_i)
                for x in pos:
                    x.calculate_fitness(self.data)
                y_r = pos[np.argmin(x_j.fit for x_j in pos)]
                dist = math.sqrt((x_i.c - predator.c)*(x_i.c - predator.c) + (x_i.gamma - predator.gamma)*(x_i.gamma - predator.gamma) + (x_i.epsilon - predator.epsilon)*(x_i.epsilon - predator.epsilon))
                sigma1 = random.random()
                sigma2 = random.random()
                l = self.l_max/(math.exp(self.beta*dist))
                x_i.c += l*sigma1*y_i_c + self.l_min*sigma2*y_r.c
                x_i.gamma += l*sigma1*y_i_gamma + self.l_min*sigma2*y_r.gamma
                x_i.epsilon += l*sigma1*y_i_epsilon + self.l_min*sigma2*y_r.epsilon
            else:
                theta = random.random()*2*math.pi
                fi = random.random()*math.pi
                aux_c = x_i.c + math.sin(fi)*math.cos(theta)
                aux_gamma = x_i.gamma + math.sin(fi)*math.sin(theta)
                aux_epsilon = x_i.epsilon + math.cos(fi)
                d1 = math.sqrt((predator.c - aux_c)*(predator.c - aux_c) + (predator.gamma - aux_gamma)*(predator.gamma - aux_gamma) + (predator.epsilon - aux_epsilon)*(predator.epsilon - aux_epsilon))
                aux_c = x_i.c - math.sin(fi)*math.cos(theta)
                aux_gamma = x_i.gamma - math.sin(fi)*math.sin(theta)
                aux_epsilon = x_i.epsilon - math.cos(fi)
                d2 = math.sqrt((predator.c - aux_c)*(predator.c - aux_c) + (predator.gamma - aux_gamma)*(predator.gamma - aux_gamma) + (predator.epsilon - aux_epsilon)*(predator.epsilon - aux_epsilon))                   
                mult = 1
                if d1 > d2:
                    mult = -1
                sigma = random.random()
                x_i.c += self.l_max*sigma*math.sin(fi)*math.cos(theta)*mult
                x_i.gamma += self.l_max*sigma*math.sin(fi)*math.sin(theta)*mult
                x_i.epsilon += self.l_max*sigma*math.cos(fi)*mult
        self.bounds(x_i)
        self.pop.append(x_i)

    def start(self):
        """Start prey-predator algorithm"""
        self.start_population()
        gen = 0
        while gen < self.max_gen:
            self.calculate_population_fitness()
            self.pop.sort(key = lambda ind: ind.fit, reverse = True)
            predator = self.pop[0]
            self.prey = self.pop
            del self.prey[0]
            print "Generation: ", gen, " Fittest: ", self.pop[-1] 
            self.pop = []
            for i in self.prey:
                self.move_prey(i, predator)
            self.move_predator(predator)
            gen += 1
        print "Final Fittest: ", self.pop[-1]
        self.pop.sort(key = lambda ind: ind.fit)
        return self.pop[0]


class GA(object):

    def __init__(self, mt_rate, cx_rate, elt_rate, pop_size, max_gen, data, params_lb, params_ub):
        self.mt_rate = mt_rate
        self.cx_rate = cx_rate
        self.elt_rate = elt_rate
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.pop = []
        self.data = data
        self.params_lb = params_lb
        self.params_ub = params_ub

    def random_ind(self):
        """Create a random individual"""
        c = random.uniform(self.params_lb[0], self.params_ub[0])
        gamma = random.uniform(self.params_lb[1], self.params_ub[1])
        epsilon = random.uniform(self.params_lb[2], self.params_ub[2])
        return SVM(c,gamma, epsilon)

    def start_population(self):
        """Start the population with random individuals"""
        for i in range(self.pop_size):
            ind = self.random_ind()
            self.pop.append(ind)

    def mutation(self,ind):
        """Mutation operator."""
        if probabilty(self.mt_rate):
            k = random.randint(0,2)
            r = random.uniform(0,1)
            if (k == 0):
                ind.c = self.params_lb[0] + r*(self.params_ub[0] - self.params_lb[0] )
            elif (k == 1):
                ind.gamma = self.params_lb[1] + r*(self.params_ub[1] - self.params_lb[1] )
            else:
                ind.epsilon = self.params_lb[2] + r*(self.params_ub[2] - self.params_lb[2] )
            ind.set_params()

    def crossover(self,ind1,ind2):
        """crossover operator. Trade lines between individuals"""
        sigma = random.uniform(0.000001,0.000009)
        new_ind1 = copy.deepcopy(ind1)
        new_ind1.fit = -1
        new_ind2 = copy.deepcopy(ind2)
        new_ind2.fit = -1
        if probabilty(self.cx_rate):
            r = random.uniform(0,1)
            if(r == 0):
                new_ind1.c = ind1.c + sigma*(ind1.c - ind2.c)
                new_ind1.gamma = ind1.gamma + sigma*(ind1.gamma - ind2.gamma)
                new_ind1.epsilon = ind1.epsilon + sigma*(ind1.epsilon - ind2.epsilon)
                new_ind2.c = ind2.c - sigma*(ind1.c - ind2.c)
                new_ind2.gamma = ind2.gamma - sigma*(ind1.gamma - ind2.gamma)
                new_ind2.epsilon = ind2.epsilon - sigma*(ind1.epsilon - ind2.epsilon)
            else:
                new_ind1.c = ind1.c + sigma*(ind2.c - ind1.c)
                new_ind1.gamma = ind1.gamma + sigma*(ind2.gamma - ind1.gamma)
                new_ind1.epsilon = ind1.epsilon + sigma*(ind2.epsilon - ind1.epsilon)
                new_ind2.c = ind2.c - sigma*(ind2.c - ind1.c)
                new_ind2.gamma = ind2.gamma - sigma*(ind2.gamma - ind1.gamma)
                new_ind2.epsilon = ind2.epsilon - sigma*(ind2.epsilon - ind1.epsilon)
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
        best_fit = 99999
        best_count = 1
        while gen < self.max_gen:
            new_population = []
            self.pop.sort(key = lambda ind: ind.fit)
            if self.pop[0].fit < best_fit:
                best_fit = self.pop[0].fit
                best_count = gen
            if(gen - best_count) > 100:
                break
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

        self.pop.sort(key = lambda ind: ind.fit)
        return self.pop[0]

def probabilty(prob):
    "Given a probabilty determines if the event is going to happen or not"
    r = random.random()
    return  r <= prob

def readCSV(file_name):
    "Read dataset from CSV file"
    data = np.loadtxt(file_name, dtype=float, delimiter=',', skiprows=1)
    scaler = preprocessing.StandardScaler()
    #data = scaler.fit_transform(data)
    np.random.shuffle(data)
    return data[:,1:], data[:,0]

def train_test(data_x, data_y):
    training_size = math.ceil(data_x.shape[0]*0.8)
    training_x = data_x[:training_size,:]
    training_y = data_y[:training_size]

    test_x = data_x[training_size:,:]
    test_y = data_y[training_size:]

    return training_x, training_y, test_x, test_y

def main():
    random.seed(argv[1])

    dataset = readCSV('data5.csv');
    data = train_test(dataset[0],dataset[1])

    params_lb = [0, 0, 1e-6]
    params_ub = [10,10,1]

    if int(argv[2]) == 0:
        print "GA:"
        mt_rate = argv[3]
        cx_rate = argv[4]

        elt_rate = 0.15
        pop_size = 100
        max_gen = 500

        ga = GA(mt_rate, cx_rate, elt_rate, pop_size, max_gen, data, params_lb, params_ub)
        best_ind = ga.start()
        best_c = best_ind.c
        best_gamma = best_ind.gamma
        best_epsilon = best_ind.epsilon
    elif int(argv[2]) == 1:
        print "PPA:"
        l_max = float(argv[3])
        l_min = float(argv[4])
        p_followup = float(argv[5])
        tau = 0.5
        beta = 1
        pop_size = 100
        max_gen  = 500
        ppa = PPA(l_min, l_max, p_followup, tau, beta, pop_size, max_gen, data, params_lb, params_ub)
        best_ind = ppa.start()
        best_c = best_ind.c
        best_gamma = best_ind.gamma
        best_epsilon = best_ind.epsilon
    else:
        best_c = float(argv[3])
        best_gamma = float(argv[4])
        best_epsilon = float(argv[5])
        

    best_svm = SVR(C=best_c, gamma=best_gamma, epsilon=best_epsilon, tol=0.00002)

    X, y = readCSV('data6.csv')
    y_sort = np.sort(y)

    kf = KFold(n_splits=5, shuffle=True)
    rmse = 0
    count = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        best_svm.fit(X_train, y_train)
        predicted = best_svm.predict(X_test)
        rmse += sqrt(mean_squared_error(y_test, predicted))
        count += 1

    print rmse/count, (rmse/count)/(y_sort[-1] - y_sort[0])

    return 0





if __name__ == "__main__":
    main()
