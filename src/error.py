from scipy.signal import freqz
import numpy as np

def minimax_error(individual, target, worN):
    w, Hi = freqz(individual, worN=worN)
    errs = []
    for wi, hi in zip(w, Hi):
        td = target(wi)
        if td is None:
            continue
        errs.append(abs(td - abs(hi))) # td is assumed to be real
    return max(errs)
