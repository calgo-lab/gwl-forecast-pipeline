import numpy as np

nse = lambda pred, real: 1 - (np.sum((pred - real) ** 2) / np.sum((real - np.mean(real)) ** 2))
nrmse = lambda pred, real: np.sqrt(np.square(pred - real).mean(axis=0)) / (np.max(real) - np.min(real))
mbe = lambda pred, real: np.mean(pred - real, axis=0)
rmbe = lambda pred, real: mbe(pred, real) / (np.max(real) - np.min(real))
