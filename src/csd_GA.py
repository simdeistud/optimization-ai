import random

from csd_utils import *
from fitness import *
from selection import *

def mut_ternary(individual, p_mut=None):
    genome = individual.genome

    if p_mut is None:
        p_mut = 1 / len(genome)

    mutated = []
    for symbol in genome:
        if random.random() < p_mut:
            choices = [-1, 0, 1]
            choices.remove(symbol)
            mutated.append(random.choice(choices))
        else:
            mutated.append(symbol)

    return CSDIndividual(genome=mutated)


def twopoint_cross(p1, p2):
    g1 = p1.genome
    g2 = p2.genome

    indexes = random.sample(range(1, len(g1) - 1), 2)
    indexes.sort()
    i, j = indexes

    c1 = g1[:i] + g2[i:j] + g1[j:]
    c2 = g2[:i] + g1[i:j] + g2[j:]

    return CSDIndividual(genome=c1), CSDIndividual(genome=c2)


def reset_broken_coeffs(individual, broken_coeffs, wordlength):
    coeffs = genomeToCoeffList(individual.genome, wordlength)

    for i in broken_coeffs:
        coeffs[i] = init_CSD_coeff(wordlength)

    return CSDIndividual(genome=coeffListToGenome(coeffs))


def make_csd_compliant_children(p1, p2, wordlength, N_digits, N_max):
    """
    Preserve the original semantics of csd_GA.py:

    1. Generate children by crossover + mutation.
    2. If a child is not CSD-compliant, regenerate that child by going again
       through crossover + mutation from the same parents.
    3. Repeat until both are compliant or N_max tries are reached.
    4. At N_max, reset only the broken coefficients of the remaining invalid children.
    """
    c1, c2 = twopoint_cross(p1, p2)
    c1 = mut_ternary(c1)
    c2 = mut_ternary(c2)

    for n in range(0, N_max + 1):
        is_c1_CSD, broken_coeffs_c1 = isCSD(c1.genome, wordlength, N_digits)
        is_c2_CSD, broken_coeffs_c2 = isCSD(c2.genome, wordlength, N_digits)

        if n == N_max:
            if not is_c1_CSD:
                c1 = reset_broken_coeffs(c1, broken_coeffs_c1, wordlength)
            if not is_c2_CSD:
                c2 = reset_broken_coeffs(c2, broken_coeffs_c2, wordlength)
            break

        if not is_c1_CSD:
            c1, _ = twopoint_cross(p1, p2)
            c1 = mut_ternary(c1)
            continue

        if not is_c2_CSD:
            _, c2 = twopoint_cross(p1, p2)
            c2 = mut_ternary(c2)
            continue

        break

    return c1, c2


def csd_GA(
    n_pop=100,
    generations=100,
    order=8,
    wordlength=8,
    N_digits=2,
    N_max=100,
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

            c1, c2 = make_csd_compliant_children(
                p1=p1,
                p2=p2,
                wordlength=wordlength,
                N_digits=N_digits,
                N_max=N_max,
            )

            newpop.append(c1)
            if len(newpop) < n_pop:
                newpop.append(c2)

        pop = newpop

    best = max(best_hist, key=fit)
    return best, best_hist