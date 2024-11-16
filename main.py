from itertools import product
from multiprocessing import Pool
import numpy as np
import config
from ga import evolve, fitness, initialize_population
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


def set_config(params):
    config.population_size = params[4]
    config.tournament_size = params[3]
    config.number_of_generations = params[5]
    config.crossover_prob = params[0]
    config.mutation_prob = params[1]
    config.gene_mutation_prob = params[2]


if __name__ == '__main__':
    data_to_use, number_of_iterations = argv[1], int(argv[2])
    data_train, labels_train, data_test, labels_test = get_data(data_to_use)

    executor = Pool(processes=config.num_processes) # Set pools to run fitness calculations in parallel

    for params in product(*config.grid):
        set_config(params)
        directory = config.get_directory_name(data_to_use, *params)
        makedirs(directory, exist_ok=True)
        print(directory)

        best_individual = None
        for i in range(number_of_iterations):
            f = open(f'{directory}{i}.txt', 'w')
            f.write(write_header())

            population = initialize_population(data_train, labels_train, executor) # Parameters are defined in config.py

            mean_fitness = 0
            std_fitness = 0
            worst_fitness = 0

            # Run the genetic algorithm
            population = evolve(population, data_train, labels_train, executor, steps=config.number_of_generations, f=f)

            if best_individual is None:
                best_individual = max(population, key=lambda x: x.fitness)
                mean_fitness = np.mean([individual.fitness for individual in population])
                std_fitness = np.std([individual.fitness for individual in population])
            else:
                best_pop_ind = max(population, key=lambda x: x.fitness)
                if best_individual.fitness < best_pop_ind.fitness:
                    best_individual = best_pop_ind
                    std_fitness = np.std([individual.fitness for individual in population])
                    mean_fitness = np.mean([individual.fitness for individual in population])
                    worst_fitness = min([individual.fitness for individual in population])
                
            
            f.close()

        best_fitness = best_individual.fitness
        fitness_in_test = fitness(best_individual, data_test, labels_test)

        bf = open(f'{directory}best.txt', 'w') 
        bf.write('best_fitness,mean_fitness,std_fitness,worst_fitness\n')
        bf.write(f'{best_fitness},{mean_fitness},{std_fitness},{worst_fitness}\n')
        bf.close()