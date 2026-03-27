from dataclasses import dataclass
from typing import Optional
import random


@dataclass
class CSDIndividual:
    genome: list[int]
    fitness: Optional[float] = None
    coeffs: Optional[list[float]] = None


def copy_individual(individual: CSDIndividual) -> CSDIndividual:
    return CSDIndividual(
        genome=individual.genome[:],
        fitness=individual.fitness,
        coeffs=None if individual.coeffs is None else individual.coeffs[:]
    )


def genomeToCoeffList(genome, wordlength):
    return [genome[i:(i + wordlength)] for i in range(0, len(genome), wordlength)]


def coeffListToGenome(coeffs):
    genome = []
    for coeff in coeffs:
        genome.extend(coeff)
    return genome


def CSDtoReal(CSDcoeff):
    r = 0.0
    for i in range(0, len(CSDcoeff)):
        r += CSDcoeff[i] * (2 ** (-i))
    return r


def genomeToRealCoeffList(genome, wordlength):
    return [CSDtoReal(coeff) for coeff in genomeToCoeffList(genome, wordlength)]


def get_coeffs(individual: CSDIndividual, wordlength: int):
    if individual.coeffs is None:
        individual.coeffs = genomeToRealCoeffList(individual.genome, wordlength)
    return individual.coeffs


def _scalarize_fitness(value):
    """
    Accept scalar fitness or a 1-element tuple/list fitness.
    This makes the code robust even if minimax_fitness() still returns (-err,)
    because of a trailing comma.
    """
    if isinstance(value, (tuple, list)):
        if len(value) != 1:
            raise ValueError(f"Fitness must be scalar or length-1 tuple/list, got: {value}")
        value = value[0]
    return float(value)


def evaluate(individual: CSDIndividual, raw_fit, wordlength: int):
    if individual.fitness is None:
        coeffs = get_coeffs(individual, wordlength)
        individual.fitness = _scalarize_fitness(raw_fit(coeffs))
    return individual.fitness


def isCSD(genome, wordlength, N_digits):
    broken_coeffs = []
    coeffs = genomeToCoeffList(genome, wordlength)

    for i in range(0, len(coeffs)):
        coeff = coeffs[i]

        if len([symbol for symbol in coeff if symbol != 0]) > N_digits:
            broken_coeffs.append(i)
            continue

        for j in range(0, len(coeff) - 1):
            if coeff[j] * coeff[j + 1] != 0:
                broken_coeffs.append(i)
                break

    return (True, []) if len(broken_coeffs) == 0 else (False, broken_coeffs)


def init_CSD_coeff(wordlength):
    coeff = [0] * wordlength
    coeff[random.randint(0, wordlength - 1)] = random.choice([-1, 1])
    return coeff


def init_individual_CSD(order, wordlength):
    genome = coeffListToGenome([init_CSD_coeff(wordlength) for _ in range(order)])
    return CSDIndividual(genome=genome)


def init_pop_CSD(N_chrom, order, wordlength):
    return [init_individual_CSD(order, wordlength) for _ in range(N_chrom)]