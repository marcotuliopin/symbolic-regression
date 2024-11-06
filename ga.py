from dataclasses import dataclass
from multiprocessing import Pool
import config
from typing import List, Optional
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import v_measure_score
from tqdm import tqdm
import grammar

@dataclass
class Individual:
    genes: np.ndarray[str]
    fitness: float = 0.0

Population = List[Individual]

@dataclass
class SelectionMethods:
    TOURNAMENT: str = 'tournament'
    ROULETTE: str = 'roulette'
    BEST: str = 'best'
    WORST: str = 'worst'


def initialize_population(data: np.ndarray, labels: np.ndarray) -> Population:
    """
    Initialize the whole population with random genes.
    """
    population = [Individual(genes=np.random.choice(['0', '1'], config.individual_size)) for _ in range(config.population_size)]

    chunks = [population[i:i + config.chunk_size] for i in range(0, len(population), config.chunk_size)]
    args_chunks = [(chunk, data, labels) for chunk in chunks]

    with Pool() as executor:
        results = executor.map(compute_fitness_chunk, args_chunks)
    population = [individual for chunk in results for individual in chunk]

    return population


def evolve(population: Population, data: np.ndarray, labels: np.ndarray, steps: int, f: str) -> Population:
    """
    Evolve the solution to the final stage.
    """
    write_stats(population, f, -1, 0)
    for i in tqdm(range(steps), desc='Progress'):
        population[:], number_of_improvements = step(population, data, labels)
        write_stats(population, f, i, number_of_improvements)


def step(population: Population, data: np.ndarray, labels: np.ndarray) -> Population:
    """
    Make one step in the evolution process.
    """
    offspring = []
    elite = select(population, SelectionMethods.BEST)
    number_of_improvements = 0

    # Crossover.
    parents = select(population, repeat=config.population_size-1)
    for _ in range(0, len(parents), 2):
        parent1, parent2 = np.random.choice(parents, 2)
        if np.random.random() < config.crossover_prob:
            children = crossover(parent1.genes, parent2.genes)
            offspring.extend(children)
        else:
            offspring.extend([parent1, parent2])
    
    # Mutation. 
    for individual in offspring:
        if np.random.random() < config.mutation_prob:
            mutate(individual)
    
    # Fitness evaluation.
    chunks = [population[i:i + config.chunk_size] for i in range(0, len(population), config.chunk_size)]
    args_chunks = [(chunk, data, labels) for chunk in chunks]
    with Pool() as executor:
        results = executor.map(compute_fitness_chunk, args_chunks)
    offspring = [individual for chunk in results for individual in chunk]

    # Calculate stats.
    mean_fitness = np.mean([individual.fitness for individual in population])
    for child in offspring:
        number_of_improvements += 1 if child.fitness > mean_fitness else 0
    
    # Elitism.
    offspring.extend(elite)

    return offspring, number_of_improvements


def select(population: Population, method: Optional[SelectionMethods]='tournament', repeat: Optional[int] = 1) -> Population:
    """
    Select one individual according to the chosen method.
    """
    match method:
        case SelectionMethods.ROULETTE:
            select = __select_by_roulette
        case SelectionMethods.TOURNAMENT:
            select = __select_by_tournament
        case SelectionMethods.BEST:
            select = __select_best
        case SelectionMethods.WORST:
            select = __select_worst
        case _:
            select = __select_by_tournament

    return [select(population) for _ in range(repeat)]


def __select_by_roulette(population: Population) -> Individual: # Only works for positive fitness
    total_fitness = sum([individual.fitness for individual in population])
    probs = [sum(individual.fitness[:i+1])/total_fitness for i, individual in enumerate(population)]

    r = np.random.random()
    for i, individual in enumerate(population):
        if r < probs[i]:
            winner = individual
            return winner


def __select_by_tournament(population: Population) -> Individual:
    tournament = np.random.choice(population, config.tournament_size, replace=False)
    winner = max(tournament, key=lambda x: x.fitness) 
    return winner


def __select_best(population: Population) -> Individual:
    return max(population, key=lambda x: x.fitness)


def __select_worst(population: Population) -> Individual:
    return min(population, key=lambda x: x.fitness)


def crossover(parent1: np.ndarray, parent2: np.ndarray) -> Individual:
    """
    Two point crossover.
    """
    lp, up = sorted(np.random.randint(0, len(parent1), 2))

    genes1 = np.concatenate([parent1[:lp], parent2[lp:up], parent1[up:]], axis=0)
    genes2 = np.concatenate([parent2[:lp], parent1[lp:up], parent2[up:]], axis=0)
    child1 = Individual(genes=genes1)
    child2 = Individual(genes=genes2)
    return [child1, child2]


def mutate(individual: Individual) -> None:
    """
    One point mutation.
    """
    mutation_mask = __select_genes(individual)
    __apply_mutation(individual, mutation_mask)


def __select_genes(individual: Individual) -> np.ndarray:
    random = np.random.random(len(individual.genes))
    mutation_mask = random < config.gene_mutation_prob
    return mutation_mask


def __apply_mutation(individual: Individual, mutation_mask: np.ndarray) -> None:
    if np.any(mutation_mask):
        individual.genes[mutation_mask] = np.random.choice(['0', '1'], size=np.sum(mutation_mask))


def fitness(individual: Individual, data: np.ndarray, labels_true: np.ndarray) -> float:
    genes = ''.join(individual.genes)
    number_of_vars = data.shape[1]
    function = translate(genes, number_of_vars)

    dmx =  _distance_matrix(data, function)
    
    if dmx is None:
        return 0

    clustering = AgglomerativeClustering(metric="precomputed", linkage="average").fit(dmx)
    labels_pred = clustering.labels_
    
    v = v_measure_score(labels_true, labels_pred)

    return v


def compute_fitness_chunk(args):
    individuals, data, labels = args
    for individual in individuals:
        individual.fitness = fitness(individual, data, labels)
    return individuals


def _distance_matrix(points, function):
    points = np.array(points)
    n = len(points)
    dist_matrix = np.zeros((n, n))
    
    # Calcula apenas o triângulo superior
    for i in range(n):
        for j in range(i + 1, n):

            try:
                diff = points[i] - points[j]
                dist = grammar.evaluate_function(function, diff)
            except (ZeroDivisionError, OverflowError):
                return None

            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist  # Espelha o valor no triângulo inferior
    
    return dist_matrix


def translate(genes, number_of_vars):
    return grammar.translate(genes, number_of_vars, config.CODON_SIZE, config.MAX_FUNCTION_SIZE)


def write_stats(population: Population, f: str, gen: int, number_of_improvements: int) -> None:
    best_individual = select(population, SelectionMethods.BEST)[0]
    worst_individual = select(population, SelectionMethods.WORST)[0]
    mean_fitness = np.mean([individual.fitness for individual in population])
    std_fitness = np.std([individual.fitness for individual in population])
    number_of_repeated_individuals = len(population) - len(set([tuple(individual.genes) for individual in population]))

    f.write(f'{gen},{best_individual.fitness},{worst_individual.fitness},{mean_fitness},{std_fitness},{number_of_repeated_individuals}, {number_of_improvements}\n')

    
if __name__ == '__main__':
    db = np.genfromtxt('dados/data/wineRed-train.csv', delimiter=',')
    data = db[:, :-1]
    labels = db[:, -1]

    a = ''.join(np.random.choice(['0', '1'], 160))
    ind = Individual(genes=np.array(list(a)))
    compute_fitness_chunk(([ind], data, labels))