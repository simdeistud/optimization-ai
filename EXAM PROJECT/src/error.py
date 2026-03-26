from scipy.signal import freqz
import numpy as np

def minimax_error(individual, target):
    w, Hi = freqz(individual)  # Hi : complex array
    Ht = np.array([target(wi) for wi in w], dtype=complex)
    diff = np.abs(Ht) - np.abs(Hi)
    return np.max(np.abs(diff))