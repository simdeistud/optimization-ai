from error import minimax_error

def embedded_fitness(fitness, target, worN):
    def embed(x):
        return fitness(x, target, worN)
    return embed

def minimax_fitness(individual, target, worN):
    return -minimax_error(individual, target, worN)