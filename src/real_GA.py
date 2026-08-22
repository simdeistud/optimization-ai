import random
from dataclasses import dataclass
from typing import Optional, Tuple
from fitness import *
from selection import *

@dataclass(slots=True)
class Individual:
    genome: Tuple[float, ...]
    fitness: Optional[float] = None


def evaluate(ind, raw_fitness):
    if ind.fitness is None:
        ind.fitness = float(raw_fitness(ind.genome))
    return ind.fitness


def init_individual(order):
    return Individual(tuple(random.uniform(-1, 1) for _ in range(order)))


def init_pop(order, n_pop):
    return [init_individual(order) for _ in range(n_pop)]


def gaussian_mut(individual, p_mut=None, sigma=0.05):
    genome = list(individual.genome)
    if p_mut is None:
        p_mut = 1 / len(genome)

    for i, coeff in enumerate(genome):
        if random.random() < p_mut:
            genome[i] = coeff + random.gauss(0, sigma)

    return Individual(tuple(genome))


def recomb_cross(p1, p2):
    child = []
    for a, b in zip(p1.genome, p2.genome):
        alpha = random.uniform(0, 1)
        child.append(alpha * a + (1 - alpha) * b)
    return Individual(tuple(child))


def real_GA(
    n_pop=100,
    generations=100,
    order=8,
    target=lambda w: 1,
    fitness=minimax_fitness,
    worN=512,
    parent_selection=roulette_pick,
    elitism=False,
    elites_perc=5,
):
    raw_fit = embedded_fitness(fitness, target, worN)
    fit = lambda ind: evaluate(ind, raw_fit)

    elites_n = int(n_pop * elites_perc / 100) if elitism else 0
    pop = init_pop(order, n_pop)
    best_hist = []

    for _ in range(generations):
        pop.sort(key=fit, reverse=True)

        elites = [Individual(ind.genome, ind.fitness) for ind in pop[:elites_n]]
        best_hist.append(Individual(pop[0].genome, pop[0].fitness))

        newpop = elites[:]
        while len(newpop) < n_pop:
            p1 = parent_selection(pop, fit)
            p2 = parent_selection(pop, fit)
            child = recomb_cross(p1, p2)
            child = gaussian_mut(child)
            newpop.append(child)

        pop = newpop

    best = max(best_hist, key=fit)
    return best, best_hist