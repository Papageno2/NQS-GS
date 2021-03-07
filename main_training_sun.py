# encoding:  utf-8

import numpy as np
import torch
import torch.nn as nn 
# from state_flip_updator import updator
# from tfim_spin1d import TFIMSpin1D, get_init_state
from state_updator import updator
from sun_spin1d import SUNSpin1D, get_init_state
from nqs_vmc1dcore_complexv2 import train 
import os
import argparse

# ----------------------- test ----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--n_sample', type=int, default=1000)
parser.add_argument('--n_optimize', type=int, default=10)
parser.add_argument('--lr',type=float, default=1E-3)
parser.add_argument('--lattice_size',type=int, default=10)
parser.add_argument('--Dp', type=int, default=2)
parser.add_argument('--threads', type=int, default=4)
parser.add_argument('--kernels', type=int, default=3)
parser.add_argument('--filters', type=int, default=4)
parser.add_argument('--layers', type=int, default=2)
parser.add_argument('--g', type=float, default=1.)
args = parser.parse_args()

Ops_args = dict(hamiltonian=SUNSpin1D, get_init_state=get_init_state, updator=updator)
Ham_args = dict(t=1, pbc=True)
net_args = dict(K=args.kernels, F=args.filters, layers=args.layers)
output_fn ='SUN_1d'

train(epochs=args.epochs, Ops_args=Ops_args, Ham_args=Ham_args, n_sample=args.n_sample, 
    n_optimize=args.n_optimize, learning_rate=args.lr, state_size=[args.lattice_size, args.Dp], 
    save_freq=10, net_args=net_args, threads=args.threads, output_fn=output_fn)
