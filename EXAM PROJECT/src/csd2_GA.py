import random

from csd_utils import *
from fitness import *
from selection import *

def mut_CSD(individual, wordlength, N_digits, p_mut=None):
    genome = individual.genome

    if p_mut is None:
        p_mut = 1 / len(genome)

    mutated = []

    for coeff in genomeToCoeffList(genome, wordlength):
        curr_d = len([symbol for symbol in coeff if symbol != 0])
        mutated_coeff = coeff[:]  # copy to avoid aliasing

        for i in range(0, len(mutated_coeff)):
            if random.random() < p_mut:
                choices = [-1, 0, 1]
                choices.remove(mutated_coeff[i])
                choice = random.choice(choices)

                if mutated_coeff[i] != 0:
                    mutated_coeff[i] = choice
                    curr_d -= 0 if choice != 0 else 1
                    continue

                if curr_d < N_digits:
                    # First symbol
                    if i == 0:
                        if mutated_coeff[i + 1] != 0:
                            mutated_coeff[i + 1] = 0
                            curr_d -= 1
                        mutated_coeff[i] = choice
                        curr_d += 1
                        continue

                    # Last symbol
                    if i == len(mutated_coeff) - 1:
                        if mutated_coeff[i - 1] != 0:
                            mutated_coeff[i - 1] = 0
                            curr_d -= 1
                        mutated_coeff[i] = choice
                        curr_d += 1
                        continue

                    # Intermediate symbols
                    if mutated_coeff[i - 1] != 0 and mutated_coeff[i + 1] != 0:
                        mutated_coeff[i - 1] = 0
                        curr_d -= 1
                        mutated_coeff[i + 1] = 0
                        curr_d -= 1
                        mutated_coeff[i] = choice
                        curr_d += 1
                        continue

                    if mutated_coeff[i - 1] != 0:
                        mutated_coeff[i - 1] = 0
                        curr_d -= 1
                        mutated_coeff[i] = choice
                        curr_d += 1
                        continue

                    if mutated_coeff[i + 1] != 0:
                        mutated_coeff[i + 1] = 0
                        curr_d -= 1
                        mutated_coeff[i] = choice
                        curr_d += 1
                        continue

                    mutated_coeff[i] = choice
                    curr_d += 1

        mutated.append(mutated_coeff)

    return CSDIndividual(genome=coeffListToGenome(mutated))


def onepoint_cross_CSD(p1, p2, wordlength):
    g1 = p1.genome
    g2 = p2.genome

    xover_point = random.randint(1, int(len(g1) / wordlength) - 1)
    xover_point *= wordlength

    c1 = g1[:xover_point] + g2[xover_point:]
    c2 = g2[:xover_point] + g1[xover_point:]

    return CSDIndividual(genome=c1), CSDIndividual(genome=c2)

def twopoint_cross_CSD(p1, p2, wordlength):
    g1 = p1.genome
    g2 = p2.genome
    indexes = random.sample(range(1, int(len(g1) / wordlength) - 1), 2)
    indexes.sort()
    i, j = indexes
    i *= wordlength
    j *= wordlength
    c1 = g1[:i] + g2[i:j] + g1[j:]
    c2 = g2[:i] + g1[i:j] + g2[j:]
    return CSDIndividual(genome=c1), CSDIndividual(genome=c2)


def csd2_GA(
    n_pop=100,
    generations=100,
    order=8,
    wordlength=8,
    N_digits=3,
    target=lambda w: 1,
    fitness=minimax_fitness,
    worN=512,
    parent_selection=roulette_pick,
    elitism=False,
    elites_perc=5,
):
    raw_fit = embedded_fitness(fitness, target, worN)
    fit = lambda ind: evaluate(ind, raw_fit, wordlength)

    elites_n = int(n_pop * elites_perc / 100) if elitism else 0
    pop = init_pop_CSD(n_pop, order, wordlength)
    best_hist = []

    for _ in range(generations):
        pop.sort(key=fit, reverse=True)

        elites = [copy_individual(ind) for ind in pop[:elites_n]]
        best_hist.append(copy_individual(pop[0]))

        newpop = elites[:]
        while len(newpop) < n_pop:
            p1 = parent_selection(pop, fit)
            p2 = parent_selection(pop, fit)

            c1, c2 = onepoint_cross_CSD(p1, p2, wordlength)
            c1 = mut_CSD(c1, wordlength, N_digits)
            c2 = mut_CSD(c2, wordlength, N_digits)

            newpop.append(c1)
            if len(newpop) < n_pop:
                newpop.append(c2)

        pop = newpop

    best = max(best_hist, key=fit)
    return best, best_hist