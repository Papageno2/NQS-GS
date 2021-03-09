# encoding: utf-8

import numpy as np
import multiprocessing
import logging
import os

def _get_unique_states(states, logphis, ustates, ucoeffs):
    """
    Returns the unique states, their coefficients and the counts.
    """
    states, indices, counts = np.unique(states, return_index=True, return_counts=True, axis=0)
    logphis = logphis[indices]
    ustates = ustates[indices]
    ucoeffs = ucoeffs[indices]
    return states, logphis, counts, ustates, ucoeffs

def _generate_updates(states, operator, single_state_shape, update_size, threads):
    """
    Generates updated states and coefficients for an Operator.

    Args:
        states: The states with shape (batch size, shape of state).
        operator: The operator used for updating the states.
        state_size: shape of a state in states
        update_size: number of update_states

    Returns:
        The updated states and their coefficients. The shape of the updated
        states is (batch size, num of updates, shape of state), where num of
        updates is the largest number of updated states among all given states.
        If a state has fewer updated states, its updates are padded with the
        original state.

    """
    n_sample = states.shape[0]
    ustates = np.zeros([n_sample, update_size] + single_state_shape)
    ucoeffs = np.zeros([n_sample, update_size])

    pool = multiprocessing.Pool(threads)
    results = []
    cnt = 0
    
    for state in states:
        results.append(pool.apply_async(operator.find_states, (state,)))
    pool.close()
    pool.join()

    for cnt, res in enumerate(results):
        ustates[cnt], ucoeffs[cnt] = res.get()

    return ustates, ucoeffs

# logger definitions
def get_logger(filename, verbosity=1, name=None):

    path = filename[0:filename.rfind("/")]
    if not os.path.isdir(path):
        os.makedirs(path)
    if not os.path.isfile(filename):
        fd = open(filename, mode="w", encoding="utf-8")
        fd.close()

    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def decimalToAny(n,x):
    # a=[0,1,2,3,4,5,6,7,8,9,'A','b','C','D','E','F']
    b=[]
    while True:
        s=n//x 
        y=n%x 
        b=b+[y]
        if s==0:
            break
        n=s
    b.reverse()

    return b