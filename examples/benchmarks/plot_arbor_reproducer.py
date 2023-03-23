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

for i, CPUS_PER_TASK in enumerate(PS_reproducer['CPUS_PER_TASK']):
    times_run = []
    NTASKS = []
    for j, N in enumerate(PS_reproducer['NTASKS']):
        NTASKS.append(N)
        pset = ParameterSet(dict(NTASKS=N,
                                 CPUS_PER_TASK=CPUS_PER_TASK,
                                 GLOBALSEED=PS_reproducer['GLOBALSEED'],
                                 POPULATION_SIZE=PS_reproducer['POPULATION_SIZE'],
                                 ))
        js = json.dumps(pset, sort_keys=True).encode()
        md5 = hashlib.md5(js).hexdigest()

        with open(os.path.join('logs', f'NTASKS_{N}_CPUS_PER_TASK_{CPUS_PER_TASK}.txt'), 'r') as f:
            times_run.append(float(f.readline()))

    label = f"CPUs per task: {CPUS_PER_TASK}"
    ax.loglog(NTASKS, times_run, ':o', label=label, base=2)
    ax.set_xticks(NTASKS)
    ax.set_xticklabels([f'{n}' for n in NTASKS])

ax.legend()
ax.set_xlabel('# MPI tasks')
ax.set_ylabel('time (ms)')

fig.savefig('PS_reproducer.pdf')

plt.show()
