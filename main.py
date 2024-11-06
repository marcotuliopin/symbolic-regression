import numpy as np
import config
from ga import evolve, initialize_population
from sys import argv
from os import makedirs


def write_header():
    return 'iteration,best,worst,mean,std,repeated_individuals,crossover_improvements\n'


def get_bc_data():
    datasets = ['breast_cancer_coimbra_train.csv', 'breast_cancer_coimbra_test.csv']
    all_data = []
    for dataset in datasets:
        db = np.genfromtxt(f'dados/data/{dataset}', delimiter=',')
        db = db[1:,:]
        raw_data = db[:, :-1]
        mean = raw_data.mean(axis=0)
        std = raw_data.std(axis=0)
        data = (raw_data - mean) / std

        labels = db[:, -1] 
        all_data.extend([data, labels])
    return all_data

def get_data(data_to_use: str):
    if data_to_use == 'bc':
        data_train, labels_train, data_test, labels_test = get_bc_data()
    return data_train, labels_train, data_test, labels_test

if __name__ == '__main__':
    data_to_use, number_of_iterations = argv[1], int(argv[2])
    data_train, labels_train, data_test, labels_test = get_data(data_to_use)

    directory = f'./experimentation/{data_to_use}/cx-{config.crossover_prob}/mx-{config.mutation_prob}/gnx-{config.gene_mutation_prob}/ts-{config.tournament_size}/pop-{config.population_size}/gen-{config.number_of_generations}/'
    makedirs(directory, exist_ok=True)
    for i in range(number_of_iterations):
        f = open(f'{directory}{i}.txt', 'w')
        f.write(write_header())

        population = initialize_population(data_train, labels_train) # Parameters are defined in config.py
        evolve(population, data_train, labels_train, steps=config.number_of_generations, f=f)

        f.close()