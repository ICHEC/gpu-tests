#!/usr/bin/env python3

"""
Here we test how to use multiple GPUs within cupy
"""

import numpy as np
import cupy as cp
from time import time
from DiagonalizationCupy import calculateEigens as get_eig

device = 'GPU'

N = 20000

numdevice = cp.cuda.runtime.getDeviceCount() # get number of GPUs



f = open("times-multi-gpu" + ".dat", 'w')
for nd in range(1, numdevice + 1):
    t1 = time()
    print(f"Running on {nd} devices")
    for ii in range(nd):
        with cp.cuda.Device(ii):
            times, e, v = get_eig(size=N, device=device)
    t2 = time()
    print(nd, t2 - t1)
    f.write(f"{nd}\t{t2 - t1}\n")
    f.flush()
#
f.close()