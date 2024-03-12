#!/usr/bin/env python3

import numpy as np
import sys
from DiagonalizationCupy import calculateEigens as get_eig

if len(sys.argv) != 2:
    print(f"USAGE: {__file__} <CPU/GPU>")
    sys.exit(1)

device = sys.argv[-1]

nsizes = np.arange(1000, 21000, 1000)

f = open("times-" + device + ".dat", 'w')
for N in nsizes:
    times, e, v = get_eig(size=N, device=device)
    print(N, times[0], times[1], times[2])
    f.write(f"{N}\t{times[0]}\t{times[1]}\t{times[2]}\n")
    f.flush()
#
f.close()