from error import minimax_error

def minimax_fitness(individual, target):
    return -minimax_error(individual, target)