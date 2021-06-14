#!/bin/bash
set -e
python --version
python -c "import numpy; print('numpy {}'.format(numpy.__version__))"
python -c "import scipy; print('scipy {}'.format(scipy.__version__))"
python -c "import neuron; print('neuron {}'.format(neuron.version))"
python -c "import LFPy; print('LFPy {}'.format(LFPy.__version__))"

python setup.py develop

while true; do
    py.test -v hybridLFPy/testing.py
    if [ $? -eq 0 ]
    then
        exit 0
        break
    fi
done
