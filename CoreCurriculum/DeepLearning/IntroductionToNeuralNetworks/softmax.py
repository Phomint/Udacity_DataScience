import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    soft = []
    exps = np.exp(L)
    soft = [e*1.0/sum(exps) for e in exps]
    return soft
