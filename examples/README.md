# hybridLFPy examples

This folder contains various examples demonstrating use of hybridLFPy to
predict LFPs from point-neuron networks.


## Files

* `example_brunel.py`: Main simulation script for predicting LFP across depth from Brunel style point-neuron network

    * `brunel_alpha_nest.py`: Brunel (2000) style point-neuron network implementation, alpha-shaped PSCs

* `example_brunel_alpha_topo_exp.py`: Compute LFP from point-neuron network with neurons positioned on a square 2D domain with exponentially decaying distance-dependent connectivity and periodic boundary conditions, alpha-shaped PSCs. The LFP is computed across lateral space using 10x10 contact sites with 400µm electrode separation (similar to Utah multi-electrode array).  

    * `brunel_alpha_topo_exp.py`: Network implementation for `example_brunel_alpha_topo_exp.py`

* `example_microcircuit.py`: LFP predictions from the Potjans & Diesmann (2014) cortical microcircuit.

    * `example_microcircuit_params.py`: Parameter file

    * `microcircuit.sli`: Network implementation in SLI

    * `binzegger_connectivity_table.json`: Layer-resolved connectivity

* `example_microcircuit*lognormal_weights.py`: Same as `example_microcircuit.py` but using connection weights drawn from lognormal distributions

* `example_plotting.py`: auxiliary codes

* `alphaisyn.mod`: NEURON NMODL file for alpha-shaped current based synapses

* `expsyni.mod`: NEURON NMODL file for exponential-shaped current based synapses

* `morphologies/`: morphology files used for LFP predictions

* `Hagen_el_al_2016_cercor/`: Main simulation codes from:

    Espen Hagen, David Dahmen, Maria L. Stavrinou, Henrik Lindén, Tom Tetzlaff,
    Sacha J van Albada, Sonja Grün, Markus Diesmann, Gaute T. Einevoll.
    "Hybrid scheme for modeling local field potentials from point-neuron networks".
    Cereb. Cortex (2016)
    DOI: 10.1093/cercor/bhw237
