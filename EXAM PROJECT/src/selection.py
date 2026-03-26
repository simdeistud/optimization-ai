import bisect
import random

def roulette_selection(pop, n_pop, fitness):
    # Compute raw fitness values
    fvals = [float(fitness(ind)) for ind in pop]

    # Shift to strictly positive domain
    min_val = min(fvals)
    shift = -min_val + 1e-12
    fvals = [f + shift for f in fvals]

    # Build cumulative distribution manually
    cum = []
    running = 0.0
    for f in fvals:
        running += f
        cum.append(running)
    tot = cum[-1]

    newpop = []
    for _ in range(n_pop):
        r = random.random() * tot
        idx = bisect.bisect_left(cum, r)

        # Clamp the index
        if idx >= len(pop):
            idx = len(pop) - 1

        newpop.append(pop[idx])

    return newpop