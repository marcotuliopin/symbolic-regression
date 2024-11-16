import numpy as np
from operator import add, sub, mul, truediv as div

fn = {
    '+': add,
    '-': sub,
    '*': mul,
    '/': div,
}

class Variable:
    @staticmethod
    def evaluate(val: int, number_of_vars: int):
        return [str(val % number_of_vars)]


class Operation:
    rules = ['+', '-', '*', '/']

    @classmethod
    def evaluate(cls, val: int):
        return [cls.rules[val % len(cls.rules)]]


class Expression:
    rules = [
        lambda: [Expression(), Operation(), Expression()],
        lambda: [Variable()],
    ]

    @classmethod
    def evaluate(cls, val: int, stop_recursion: bool = False):
        if stop_recursion: # Stop recursion if the function is too big.
            return cls.rules[1]()
        return cls.rules[val % len(cls.rules)]()


def translate(genes: str, number_of_vars: int, codon_size: int, max_function_size: int):
    function = [Expression()]
    function_idx = 0
    genes_idx = 0
    number_of_codons = len(genes) // codon_size

    depths = [0]

    while function_idx < len(function):
        while function_idx < len(function) and is_terminal(function[function_idx]):
            function_idx += 1

        if function_idx >= len(function):
            break

        codon = int(genes[genes_idx * codon_size: (genes_idx + 1) * codon_size], 2)

        current_depth = depths[function_idx]

        if isinstance(function[function_idx], Variable):
            val = function[function_idx].evaluate(codon, number_of_vars)

        elif isinstance(function[function_idx], Expression):
            val = function[function_idx].evaluate(codon, current_depth >= max_function_size)
            
            new_depth = current_depth + 1
            depths[function_idx:function_idx + 1] = [new_depth] * len(val)

        else:
            val = function[function_idx].evaluate(codon)

        function[function_idx:function_idx + 1] = val

        genes_idx = (genes_idx + 1) % number_of_codons

    return function
    

def evaluate_function(expr: np.ndarray, diffs: np.ndarray):
    if len(expr) == 1:
        return diffs[int(expr[0])]
    
    left = diffs[int(expr[0])]
    op = expr[1]
    right = diffs[int(expr[2])]
    if op == '/' and right == 0:
        right = 1

    result = fn[op](left, right)

    i = 3
    while i < len(expr):
        op = expr[i]
        right = diffs[int(expr[i + 1])]
        if op == '/' and right == 0:
            right = 1

        result = fn[op](result, right)
        i += 2

    return result


def is_terminal(val):
    return isinstance(val, str)
    
    
if __name__ == '__main__':
    a = ''.join(np.random.choice(['0', '1'], 160))
    CODON_SIZE = 16
    MAX_FUNCTION_SIZE = 100
    NUM_VARIABLES = 10
    f = translate(a, NUM_VARIABLES, CODON_SIZE, MAX_FUNCTION_SIZE)