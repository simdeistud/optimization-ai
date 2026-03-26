import random
from fitness import *
from selection import *

def init_individual(order):
    return [random.uniform(-1, 1) for _ in range(order)]

def init_pop(order, n_pop):
    return [init_individual(order) for _ in range(n_pop)]

def gaussian_mut(individual, p_mut=None, sigma=0.05):
    if p_mut is None:
        p_mut = 1 / len(individual)

    mutated = []
    for coeff in individual:
        if random.random() < p_mut:
            mutated.append(coeff + random.gauss(0, sigma))
        else:
            mutated.append(coeff)
    return mutated

def recomb_cross(p1, p2):
    child = []
    for a, b in zip(p1, p2):
        alpha = random.uniform(0, 1)
        child.append(alpha * a + (1 - alpha) * b)
    return child

def real_GA(
    n_pop=100,
    generations=100,
    order=8,
    target=lambda w: 1,
    fitness=minimax_fitness,
    worN=512,
    parent_selection=roulette_pick,
    elitism=True,
    elites_perc=5,
    tournament_k=3
):
    fit = embedded_fitness(fitness, target, worN)

    elites_n = int(n_pop * elites_perc / 100) if elitism else 0
    pop = init_pop(order, n_pop)
    best_hist = []

    for _ in range(generations):
        # sort by fitness (higher is better, since minimax_fitness = -error)
        pop.sort(key=fit, reverse=True)

        # keep elites
        elites = [ind[:] for ind in pop[:elites_n]]
        best_hist.append(pop[0][:])

        # build next generation
        newpop = elites[:]

        while len(newpop) < n_pop:
            if parent_selection == tournament_pick:
                p1 = tournament_pick(pop, fit, k=tournament_k)
                p2 = tournament_pick(pop, fit, k=tournament_k)
            else:
                p1 = roulette_pick(pop, fit)
                p2 = roulette_pick(pop, fit)

            child = recomb_cross(p1, p2)
            child = gaussian_mut(child)
            newpop.append(child)

        pop = newpop

    best = max(best_hist, key=fit)
    return best, best_hist