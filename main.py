import numpy as np
import matplotlib.pyplot as plt
from ga import SelectionMethods, evolve, initialize_population, select

db = np.genfromtxt('dados/data/wineRed-train.csv', delimiter=',')
data = db[:, :-1]
labels = db[:, -1]

print("Initializing...")
population = initialize_population(data, labels) # Parameters are defined in ga.py.
print("Initial population: ", len(population))
evolve(population, data, labels, steps=1000)

best_individual = select(population, SelectionMethods.BEST)
print(best_individual.fitness)