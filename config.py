population_size: int = 50
tournament_size: int = 2
number_of_generations: int = 50
crossover_prob: float = 0.9
mutation_prob: float = 0.05
gene_mutation_prob: float = 0.05

num_processes: int = 20

CODON_SIZE = 8
MAX_FUNCTION_SIZE = 7
chunk_size = population_size // num_processes
individual_size: int = CODON_SIZE * 10

grid = [
    # [0.7, 0.9], # crossover_prob
    [0.9], # crossover_prob
    [0.05, 0.1, 0.3, 0.5], # mutation_prob
    # [0.05], # mutation_prob
    # [0.05, 0.1, 0.3, 0.5], # gene_mutation_prob
    [0.5], # gene_mutation_prob
    # [2, 4, 5], # tournament_size
    [2], # tournament_size
    # [20, 50, 100, 500], # population_size
    [300], # population_size
    # [30, 50, 100, 300, 500], # number_of_generations
    [300], # number_of_generations
]

def get_directory_name(data_to_use, cx, mx, gnx, ts, pop, gen):
    return f'./experimentation/{data_to_use}/cx-{cx}/mx-{mx}/gnx-{gnx}/ts-{ts}/pop-{pop}/gen-{gen}/'