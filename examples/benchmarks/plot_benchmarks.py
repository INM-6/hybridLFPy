import os
import numpy as np
import hashlib
import json
from parameters import ParameterSpace, ParameterSet
import matplotlib.pyplot as plt

import brunel_alpha_nest as BN


PS0 = ParameterSpace('PS0.txt')
keys = PS0.range_keys()

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.set_title(f'# neurons: {}; host: {}'os.environ['HOST'])

for i, SIM_SCRIPT in enumerate(PS0['SIM_SCRIPT']):
    times_pop = []
    times_run = []
    NTASKS = []
    for j, N in enumerate(PS0['NTASKS']):
        NTASKS.append(N)
        pset = ParameterSet(dict(NTASKS=N,
                                 SIM_SCRIPT=SIM_SCRIPT,
                                 GLOBALSEED=PS0['GLOBALSEED']))
        js = json.dumps(pset, sort_keys=True).encode()
        md5 = hashlib.md5(js).hexdigest()

        with open(os.path.join(md5, 'time_pop.txt'), 'r') as f:
            times_pop.append(float(f.readline()))

        with open(os.path.join(md5, 'time_run.txt'), 'r') as f:
            times_run.append(float(f.readline()))

    label = f"{'LFPy' if SIM_SCRIPT == 'example_brunel.py' else 'Arbor'} pop"
    ax.loglog(NTASKS, times_pop, '-o', label=label, base=2)

    label = f"{'LFPy' if SIM_SCRIPT == 'example_brunel.py' else 'Arbor'} run"
    ax.loglog(NTASKS, times_run, ':o', label=label, base=2)

ax.legend()
ax.set_xlabel('NTASKS')
ax.set_ylabel('time (ms)')
