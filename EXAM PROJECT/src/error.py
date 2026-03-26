from scipy.signal import freqz
import numpy as np

def minimax_error(individual, target, worN):
    w, Hi = freqz(individual, worN=worN)  # Hi : complex array
    Ht = np.array([target(wi) for wi in w], dtype=complex)
    diff = np.abs(Ht) - np.abs(Hi)
    return np.max(np.abs(diff))