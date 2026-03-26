import random

def genomeToCoeffList(genome, wordlength):
    return [genome[i:(i + wordlength)] for i in range(0, len(genome), wordlength)]

def coeffListToGenome(coeffs):
    genome = []
    for coeff in coeffs:
        genome.extend(coeff)
    return genome

def CSDtoReal(CSDcoeff):
    r = 0
    for i in range(0, len(CSDcoeff)):
        r += CSDcoeff[i]*(2 ** (-i))
    return r

def genomeToRealCoeffList(genome, wordlength):
    return [CSDtoReal(coeff) for coeff in genomeToCoeffList(genome, wordlength)]

def isCSD(genome, wordlength, N_digits):
    broken_coeffs = []
    coeffs = genomeToCoeffList(genome, wordlength)
    for i in range (0, len(coeffs)):
        coeff = coeffs[i]
        if len([symbol for symbol in coeff if symbol not in [0]]) > N_digits:
            broken_coeffs.append(i)
            continue
        for j in range (0, len(coeff)-1):
            if coeff[j]*coeff[j+1] != 0:
                broken_coeffs.append(i)
                break
    return (True, []) if len(broken_coeffs) == 0 else (False, broken_coeffs)

def init_CSD_coeff(wordlength):
    coeff = [0] * wordlength
    coeff[random.randint(0, wordlength-1)] = random.choice([-1, 1])
    return coeff

def init_individual_CSD(order, wordlength):
    return coeffListToGenome([init_CSD_coeff(wordlength) for _ in range(0, order)])

def init_pop_CSD(N_chrom, order, wordlength):
    return [init_individual_CSD(order, wordlength) for _ in range(N_chrom)]


def csd_fitness(fitness, wordlength, target, worN):
    def csd(x):
        return fitness(genomeToRealCoeffList(x, wordlength), target, worN)
    return csd
