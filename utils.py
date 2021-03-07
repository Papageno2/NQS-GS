# encoding: utf-8

import numpy as np
import logging
import os

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