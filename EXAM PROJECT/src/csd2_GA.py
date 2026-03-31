import random

from csd_utils import *
from fitness import *
from selection import *

def mut_CSD(individual, wordlength, N_digits, p_mut=None):
    """
    Perform CSD-aware mutation directly on the flat genome.

    Mutation respects:
        - canonical CSD (no adjacent ±1)
        - maximum number of non-zero digits per coefficient (N_digits)

    Returns
    -------
    CSDIndividual
        A new mutated individual.
    """
    genome = individual.genome

    if p_mut is None:
        p_mut = 1 / len(genome)

    rand = random.random
    pick = random.choice

    ALT = {
        -1: (0, 1),
         0: (-1, 1),
         1: (-1, 0),
    }

    # Flat mutated genome, same structure as input
    mutated_genome = genome[:]

    # Process one coefficient at a time, but without converting the full genome
    for start in range(0, len(genome), wordlength):
        end = start + wordlength
        word = mutated_genome[start:end]   # local copy of one coefficient
        n = len(word)

        # Track nonzero positions inside this coefficient
        nonzero = {i for i, v in enumerate(word) if v != 0}

        for i in range(n):
            if rand() >= p_mut:
                continue

            old = word[i]
            new = pick(ALT[old])

            # Case 1: mutate an existing nonzero digit
            if old != 0:
                word[i] = new
                if new == 0:
                    nonzero.remove(i)
                continue

            # Case 2: insert ±1 into a zero digit

            # Enforce canonical CSD: clear adjacent nonzeros
            for j in (i - 1, i + 1):
                if n > j >= 0 != word[j]:
                    word[j] = 0
                    nonzero.remove(j)

            # Enforce max nonzero-digit budget
            if len(nonzero) == N_digits:
                lsnz = max(nonzero)   # least significant nonzero index
                word[lsnz] = 0
                nonzero.remove(lsnz)

            word[i] = new
            nonzero.add(i)

        # Write the mutated coefficient back into the flat genome
        mutated_genome[start:end] = word

    if not isCSD(mutated_genome, wordlength, N_digits):
        print("ERROR!")

    return CSDIndividual(genome=mutated_genome)


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
    N_digits=2,
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

            c1, c2 = twopoint_cross_CSD(p1, p2, wordlength)
            c1 = mut_CSD(c1, wordlength, N_digits)
            c2 = mut_CSD(c2, wordlength, N_digits)

            newpop.append(c1)
            if len(newpop) < n_pop:
                newpop.append(c2)

        pop = newpop

    best = max(best_hist, key=fit)
    return best, best_hist