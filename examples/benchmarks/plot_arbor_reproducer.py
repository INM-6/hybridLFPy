#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import hashlib
import json
from parameters import ParameterSpace, ParameterSet
import matplotlib.pyplot as plt


PS_reproducer = ParameterSpace('PS_reproducer.txt')
keys = PS_reproducer.range_keys()

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
for h in ['HOST', 'HOSTNAME']:
    if h in os.environ:
        host = os.environ[h]
        break

# ax.set_title(f"network size: {5 * BN.order}; host: {host}")

for i, NTHREADS in enumerate(PS_reproducer['NTHREADS']):
    times_run = []
    NTASKS = []
    for j, N in enumerate(PS_reproducer['NTASKS']):
        NTASKS.append(N)
        pset = ParameterSet(dict(NTASKS=N,
                                 NTHREADS=NTHREADS,
                                 GLOBALSEED=PS_reproducer['GLOBALSEED'],
                                 POPULATION_SIZE=PS_reproducer['POPULATION_SIZE'],
                                 ))
        js = json.dumps(pset, sort_keys=True).encode()
        md5 = hashlib.md5(js).hexdigest()

        with open(os.path.join('logs', f'MPI_SIZE_{N}_NTHREADS_{NTHREADS}.txt'), 'r') as f:
            times_run.append(float(f.readline()))

    label = f"nthreads: {NTHREADS}"
    ax.loglog(NTASKS, times_run, ':o', label=label, base=2)
    ax.set_xticks(NTASKS)
    ax.set_xticklabels([f'{n}' for n in NTASKS])

ax.legend()
ax.set_xlabel('# MPI processes"')
ax.set_ylabel('time (ms)')

fig.savefig('PS_reproducer.pdf')

plt.show()
