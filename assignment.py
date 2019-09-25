import numpy as np
from numbers import Number
from random import shuffle, seed
from functools import reduce
from operator import add


def clean_weights(weights):
    if isinstance(weights, Number):
        weights = [weights]
    if sum(weights) < 1:
        weights = [1 - sum(weights)] + weights
    return weights


def get_assignments_as_positions(assignment):
    #assignment = np.array(assignment)
    return [np.where(np.array(assignment) == i)[0]
            for i in range(max(assignment) + 1)]


def draw_iid_assignment(weights, sample_size):
    weights = clean_weights(weights)
    return np.random.choice(
        range(len(weights)), size=sample_size, replace=True, p=weights)


def draw_shuffled_assignment(weights, sample_size):
    weights = clean_weights(weights)
    treatment_list = [int(np.ceil(w * sample_size)) * [i]
                      for i, w in enumerate(weights)]
    assignment = reduce(add, treatment_list, [])
    shuffle(assignment)
    return assignment
