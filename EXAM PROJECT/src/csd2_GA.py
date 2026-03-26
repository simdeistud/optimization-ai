import random
from csd_utils import *
from fitness import *
from selection import *

def mut_CSD(individual, wordlength, N_digits, p_mut=None):
    if p_mut is None:
        p_mut = 1/wordlength
    mutated = []
    for coeff in genomeToCoeffList(individual, wordlength):
        curr_d = len([symbol for symbol in coeff if symbol not in [0]])
        mutated_coeff = coeff

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
                    # Special case for first symbol
                    if i == 0:
                        if mutated_coeff[i+1] != 0:
                            mutated_coeff[i+1] = 0
                            curr_d -= 1
                        mutated_coeff[i] = choice
                        curr_d += 1
                        continue
                    # Special case for last symbol
                    if i == len(mutated_coeff)-1:
                        if mutated_coeff[i-1] != 0:
                            mutated_coeff[i-1] = 0
                            curr_d -= 1
                        mutated_coeff[i] = choice
                        curr_d += 1
                        continue
                    # Intermediate symbols
                    if mutated_coeff[i-1] != 0 and mutated_coeff[i+1] != 0:
                        mutated_coeff[i-1] = 0
                        curr_d -= 1
                        mutated_coeff[i+1] = 0
                        curr_d -= 1
                        mutated_coeff[i] = choice
                        curr_d += 1
                        continue
                    if mutated_coeff[i-1] != 0:
                        mutated_coeff[i-1] = 0
                        curr_d -= 1
                        mutated_coeff[i] = choice
                        curr_d += 1
                        continue
                    if mutated_coeff[i+1] != 0:
                        mutated_coeff[i+1] = 0
                        curr_d -= 1
                        mutated_coeff[i] = choice
                        curr_d += 1
                        continue
                    mutated_coeff[i] = choice
                    curr_d += 1
        mutated.append(mutated_coeff)
    return coeffListToGenome(mutated)

def onepoint_cross_CSD(p1, p2, wordlength):
    xover_point = random.randint(1, int(len(p1)/wordlength) - 1)
    xover_point *= wordlength
    return p1[:xover_point] + p2[xover_point:], p2[:xover_point] + p1[xover_point:]

def GA_CSD2(
        n_pop=100,
        generations=100,
        order=8,
        wordlength=8,
        N_digits=3,
        target=lambda w: 1,
        fitness=minimax_fitness,
        worN=512,
        selection=roulette_selection,
        elitism=False,
        elites_perc=5):
    best = []
    N_elites = int(n_pop / 100 * elites_perc) if elitism else 0

    # Population initialization
    pop = init_pop_CSD(n_pop, order, wordlength)

    for _ in range(generations):
        pop.sort(csd_fitness(fitness, wordlength, target, worN), reverse=True)
        elites = pop[:N_elites]
        best.append(pop[0])
        for _ in range(n_pop):
            p1 = pop[random.randint(0, len(pop) - 1)]
            p2 = pop[random.randint(0, len(pop) - 1)]
            c1, c2 = onepoint_cross_CSD(p1, p2, wordlength)
            c1, c2 = mut_CSD(c1, wordlength, N_digits), mut_CSD(c2, wordlength, N_digits)
            # OPTIONAL: uncomment to check if all individuals are CSD
            #if not isCSD(c1, wordlength, N_digits)[0] or not isCSD(c2, wordlength, N_digits)[0]:
            #    print(f"c1: {isCSD(c1, wordlength, N_digits)[0]}")
            #    print(f"c2: {isCSD(c2, wordlength, N_digits)[0]}")
            pop.extend([c1, c2])
        pop = selection(pop, n_pop - N_elites, csd_fitness(fitness, wordlength, target, worN))
        pop.extend(elites)
    if elitism: best.append(max(best, csd_fitness(fitness, wordlength, target, worN)))
    return best