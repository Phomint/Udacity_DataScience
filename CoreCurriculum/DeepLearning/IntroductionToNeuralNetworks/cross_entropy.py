import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    sigma = 0
    for i in range(len(P)):
        sigma = sigma-(Y[i]*np.log(P[i])+(1-Y[i])*np.log(1-P[i]))
    return sigma
