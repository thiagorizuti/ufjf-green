import operator
import math
import random
import csv
import itertools
import numpy
from datetime import datetime
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from sys import argv
from sklearn import  preprocessing

def scaleData(data):
    scaler = preprocessing.MinMaxScaler()
    return scaler.fit_transform(data).tolist()


random.seed(10)
with open("data4.csv") as dataset:
    reader = csv.reader(dataset)
    dataset = list(list(float(elem) for elem in row) for row in reader)
dataset = scaleData(dataset)

att = 16
test = []
random.shuffle(dataset)

for i in range(int(len(dataset)*0.9)):
    x = dataset.pop(0)
    test.append(x)

def safeDiv(a, b):
    if b == 0:
        return 0
    return a / b

pset = gp.PrimitiveSetTyped("main", itertools.repeat(float, att),float)
pset.addPrimitive(operator.lt, [float, float], bool)
pset.addPrimitive(operator.le, [float, float], bool)
pset.addPrimitive(operator.gt, [float, float], bool)
pset.addPrimitive(operator.ge, [float, float], bool)
pset.addPrimitive(operator.eq, [float, float], bool)
pset.addPrimitive(operator.ne, [float, float], bool)
pset.addPrimitive(operator.xor, [bool, bool], bool)
pset.addPrimitive(operator.and_, [bool, bool], bool)
pset.addPrimitive(operator.or_, [bool, bool], bool)
pset.addPrimitive(operator.not_, [bool], bool)
pset.addPrimitive(operator.add, [float,float], float)
pset.addPrimitive(operator.sub, [float,float], float)
pset.addPrimitive(operator.mul, [float,float], float)
pset.addPrimitive(safeDiv, [float,float], float)
pset.addPrimitive(math.cos, [float], float)
pset.addPrimitive(math.sin, [float], float)


creator.create("FitnessMulti", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def eval(individual):
    func = toolbox.compile(expr=individual)
    sqerrors = []
    for i in range(len(dataset)):
        #print func(*dataset[i][1:])
        sqerrors.append((func(*dataset[i][1:]) - dataset[i][0])**2)
    #sqerrors = ((func(*dataset[i][1:]) - dataset[i][0])**2 for i in range(len(dataset)))
    return math.fsum(sqerrors) / len(dataset),


toolbox.register("evaluate", eval)
toolbox.register("select", tools.selRoulette)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genGrow, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(operator.attrgetter('height'),max_value=15))

def accuracy(individual, dataset):
    func = toolbox.compile(expr=individual)
    sqerrors = []
    for i in range(len(dataset)):
        sqerrors.append((func(*dataset[i][1:]) - dataset[i][0])**2)
        print func(*dataset[i][1:]), dataset[i][0]

    return math.fsum(sqerrors) / len(dataset),


def main():
    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    cxpb = 0.9
    mutpb = 0.2
    ngen = 100
    algorithms.eaSimple(pop, toolbox, cxpb, mutpb,ngen, halloffame=hof,verbose=True)
    for ind in hof:
        ind1 = ind
        print ind
        print accuracy(ind,test)
        print " "


    return 0


if __name__ == "__main__":
    main()
