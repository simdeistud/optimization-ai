import random
from fitness import minimax_fitness
from selection import roulette_selection


def init_individual(order):
    individual = []
    for _ in range(order):
        individual.append(random.uniform(-1, 1))
    return individual


def init_pop(order, n_pop):
    return [init_individual(order) for _ in range(n_pop)]


def gaussian_mut(individual, p=0.01):
    mutated = []
    for coeff in individual:
        if random.random() < p:
            mutated.append(coeff + random.gauss())
        else:
            mutated.append(coeff)
    return mutated


def recomb_cross(p1, p2):
    child = []
    for p, q in zip(p1, p2):
        a = random.uniform(0, 1)
        child.append(a * p + (1 - a) * q)
    return child


def real_GA(
        n_pop=100,
        generations=100,
        order=8,
        target=lambda w: 1,
        fitness=minimax_fitness,
        selection=roulette_selection,
        elitism=False,
        elites_perc=5):
    best = []
    elites_n = int(n_pop / 100 * elites_perc) if elitism else 0
    # Population initialization
    pop = init_pop(order, n_pop)
    for _ in range(generations):
        pop.sort(key=lambda x: fitness(x, target), reverse=True)
        elites = pop[:elites_n]
        best.append(pop[0])
        for _ in range(n_pop):
            parent1 = pop[random.randint(0, len(pop) - 1)]
            parent2 = pop[random.randint(0, len(pop) - 1)]
            child = gaussian_mut(recomb_cross(parent1, parent2))
            pop.append(child)
        pop = selection(pop, n_pop - elites_n, fitness, target)
        pop.extend(elites)
    if elitism: best.append(max(best, key=lambda x: fitness(x, target)))
    return best
