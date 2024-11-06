import numpy as np
from operator import add, sub, mul, truediv as div

precedence = {
    '*': 2,
    '/': 2,
    '+': 1,
    '-': 1,
}

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

    while function_idx < len(function):
        while function_idx < len(function) and is_terminal(function[function_idx]): # Don't evaluate terminals.
            function_idx += 1

        if function_idx >= len(function):
            break

        codon = int(genes[genes_idx * 8: (genes_idx + 1) * 8], 2)

        if isinstance(function[function_idx], Variable):
            val = function[function_idx].evaluate(codon, number_of_vars)

        elif isinstance(function[function_idx], Expression):
            val = function[function_idx].evaluate(codon, len(function) >= max_function_size)

        else:
            val = function[function_idx].evaluate(codon)

        function[function_idx: function_idx + 1] = val # Replace the current function with the evaluated one.
        genes_idx = (genes_idx + 1) % number_of_codons

    return ''.join(function)
    

def evaluate_function(expr, diffs):
    stack = []
    operations = []
    
    for token in expr:
        if not token in precedence: # If its a variable.
            stack.append(diffs[int(token)])

        else:
            # When an operation with less precedence is found, we evaluate the stack before
            # adding the new operation to the stack.
            while (operations and precedence[operations[-1]] >= precedence[token]):
                right = stack.pop()
                left = stack.pop()
                op = operations.pop()

                if op == '/' and right == 0:
                    right = 1 # Avoid division by zero.

                result = fn[op](left, right)
                stack.append(result) # The result is pushed back to the stack.

            operations.append(token)
    # Evaluate the remaining operations in the stack.
    while operations:
        right = stack.pop()
        left = stack.pop()
        op = operations.pop()
        

        if op == '/' and right == 0:
            right = 1 # Avoid division by zero.

        result = fn[op](left, right)
        stack.append(result)
    
    return stack[0] 


def is_terminal(val):
    return isinstance(val, str)
    
    
if __name__ == '__main__':
    a = ''.join(np.random.choice(['0', '1'], 160))
    CODON_SIZE = 16
    MAX_FUNCTION_SIZE = 100
    NUM_VARIABLES = 10
    f = translate(a, NUM_VARIABLES, CODON_SIZE, MAX_FUNCTION_SIZE)
    print('-' in precedence)