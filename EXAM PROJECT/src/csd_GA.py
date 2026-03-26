import random
from csd_utils import *
from fitness import *
from selection import *

def mut_ternary(individual, p_mut=None):
    if p_mut is None:
        p_mut = 1/len(individual)
    mutated = []
    for symbol in individual:
        if random.random() < p_mut:
            choices = [-1, 0, 1]
            choices.remove(symbol)
            mutated.append(random.choice(choices))
        else:
            mutated.append(symbol)
    return mutated

def twopoint_cross(p1, p2):
    indexes = random.sample(range(1, len(p1) - 1), 2)
    indexes.sort()
    return p1[:indexes[0]] + p2[indexes[0]:indexes[1]] + p1[indexes[1]:], p2[:indexes[0]] + p1[indexes[0]:indexes[1]] + p2[indexes[1]:]

def csd_GA(
        n_pop=100,
        generations=100,
        order=8,
        wordlength=8,
        N_digits=3,
        N_max=100,
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
            c1, c2 = twopoint_cross(p1, p2)
            c1 = mut_ternary(c1)
            c2 = mut_ternary(c2)
            for n in range(0, N_max+1):
                is_c1_CSD, broken_coeffs_c1 = isCSD(c1, wordlength, N_digits)
                is_c2_CSD, broken_coeffs_c2 = isCSD(c2, wordlength, N_digits)
                if n == N_max:
                    c1_fixed = genomeToCoeffList(c1, wordlength)
                    c2_fixed = genomeToCoeffList(c2, wordlength)
                    for i in broken_coeffs_c1:
                        c1_fixed[i] = init_CSD_coeff(wordlength)
                    for i in broken_coeffs_c2:
                        c2_fixed[i] = init_CSD_coeff(wordlength)
                    c1 = coeffListToGenome(c1_fixed)
                    c2 = coeffListToGenome(c2_fixed)
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
            pop.extend([c1, c2])
        pop = selection(pop, n_pop - N_elites, csd_fitness(fitness, wordlength, target, worN))
        pop.extend(elites)
    if elitism: best.append(max(best, csd_fitness(fitness, wordlength, target, worN)))
    return best