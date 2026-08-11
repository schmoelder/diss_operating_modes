# Operating modes study

This repository contains the code used for the dissertation study on advanced chromatographic operating modes.
The case studies and their results are described in Chapter 6 of the dissertation.
The dissertation source is available in the thesis repository:
[diss](https://github.com/schmoelder/diss)
In the thesis working tree, Chapter 6 lives under `doc/06_operating_modes/` and is the authoritative narrative description of the operating modes, parameter choices, optimization problems, and reported figures.

The output repository for CADET-RDM result branches is available here:
[diss_operating_modes_output](https://github.com/schmoelder/diss_operating_modes_output)


## Reproducing the study

Create a fresh environment from the pinned specification and run the full study from the package directory:

```bash
git clone https://github.com/schmoelder/diss_operating_modes.git
cd diss_operating_modes
conda env create -f environment.yml
conda activate operating_modes
cd operating_modes
python run_all.py
```

Runs with push access to the output repository can publish CADET-RDM result branches automatically.
Without push access, the calculations can still be reproduced locally from the pinned environment and source state, but result branches remain local or pushing must be disabled.

The scripted reproduction follows the model choices used for the reported study.
Study variants, cached result branch names, and post-processing choices are encoded in `run_all.py` and the modules under `operating_modes/`.
