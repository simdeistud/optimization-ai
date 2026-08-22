import random
import bisect
import math

def roulette_pick(pop, fitness):
    fvals = [float(fitness(ind)) for ind in pop]

    # shift because fitness may be negative
    min_val = min(fvals)
    weights = [f - min_val + 1e-12 for f in fvals]

    total = sum(weights)

    # fallback if all weights collapse
    if not math.isfinite(total) or total <= 0:
        return random.choice(pop)

    cum = []
    running = 0.0
    for w in weights:
        running += w
        cum.append(running)

    r = random.random() * total
    idx = bisect.bisect_left(cum, r)
    if idx >= len(pop):
        idx = len(pop) - 1
    return pop[idx]


def tournament_pick(pop, fitness, k=3):
    contestants = random.choices(pop, k=k)
    return max(contestants, key=fitness)