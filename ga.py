from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import v_measure_score
from tqdm import tqdm
import grammar

population_size: int = 300
tournament_size: int = 5
crossover_prob: float = 0.8
mutation_prob: float = 0.5

CODON_SIZE = 8
MAX_FUNCTION_SIZE = 20
individual_size: int = CODON_SIZE * 10
gene_mutation_prob: float = 0.04


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


def evolve(population: Population, data: np.ndarray, labels: np.ndarray, steps: int) -> Population:
    """
    Evolve the solution to the final stage.
    """
    for _ in tqdm(range(steps), desc='Progress'):
        population[:] = step(population, data, labels)


def step(population: Population, data: np.ndarray, labels: np.ndarray) -> Population:
    """
    Make one step in the evolution process.
    """
    offspring = select(population, repeat=population_size-1)
    elite = select(population, SelectionMethods.BEST)

    print("Crossover")
    for _ in range(0, len(offspring), 2):
        parent1, parent2 = np.random.choice(offspring, 2)
        if np.random.random() < crossover_prob:
            offspring.extend(crossover(parent1.genes, parent2.genes))
        else:
            offspring.extend([parent1, parent2])
    
    print("Mutation")
    for individual in offspring:
        if np.random.random() < mutation_prob:
            mutate(individual)
    
    print("Fitness")
    for individual in offspring:
        individual.fitness = fitness(individual, data, labels)
    
    offspring.extend(elite)
    print('Offspring length: ', len(offspring))

    return offspring


def initialize_population(data: np.ndarray, labels: np.ndarray) -> Population:
    """
    Initialize the whole population with random genes.
    """
    population = [Individual(genes=np.random.choice(['0', '1'], individual_size)) for _ in range(population_size)]
    for individual in population:
        individual.fitness = fitness(individual, data, labels)
    return population


def select(population: Population, method: Optional[SelectionMethods]='tournament', repeat: Optional[int] = 1) -> Population:
    """
    Select one individual according to the chosen method.
    """
    if method == SelectionMethods.ROULETTE:
        select = __select_by_roulette
    elif method == SelectionMethods.TOURNAMENT:
        select = __select_by_tournament
    else:
        select = __select_best

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
    tournament = np.random.choice(population, tournament_size, replace=False)
    winner = max(tournament, key=lambda x: x.fitness) 
    return winner


def __select_best(population: Population) -> Individual:
    return max(population, key=lambda x: x.fitness)


def crossover(parent1: np.ndarray, parent2: np.ndarray) -> Individual:
    """
    Two point crossover.
    """
    lp, up = np.random.randint(0, len(parent1), 2)
    if lp > up:
        lp, up = up, lp

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
    mutation_mask = random < mutation_prob
    return mutation_mask


def __apply_mutation(individual: Individual, mutation_mask: np.ndarray) -> None:
    if np.any(mutation_mask):
        individual.genes[mutation_mask] = np.random.choice(['0', '1'], size=np.sum(mutation_mask))


def fitness(individual: Individual, data: np.ndarray, labels_true: np.ndarray) -> float:
    genes = ''.join(individual.genes)
    number_of_vars = data.shape[1]
    function = grammar.translate(genes, number_of_vars, CODON_SIZE, MAX_FUNCTION_SIZE)

    print(function)
    
    dmx =  _distance_matrix(data, function)
    clustering = AgglomerativeClustering(metric="precomputed", linkage="average").fit(dmx)
    labels_pred = clustering.labels_
    
    v = v_measure_score(labels_true, labels_pred)

    return v

def _distance_matrix(points, expr):
    n = len(points)
    dist_matrix = np.zeros((n, n))
    
    # Calcula apenas o triângulo superior
    for i in range(n):
        for j in range(i + 1, n):
            dist = grammar.evaluate_function(expr, points[i] - points[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist  # Espelha o valor no triângulo inferior
    
    return dist_matrix
