# dadi_DFE_inference
Python scripts which carry out demographic and DFE inference on tree sequences simulated by SLiM. 

These are split into two folders: demographic_inference and DFE_inference_and_caches. 

### demographic_inference
This folder contains scripts and additional files for inferring a demographic models from tree sequences simulated by SLiM. 

The files are:

- dadi_demographic_inference.py: Demographic inference script for dadi and msprime/tskit code for building the SFS from tree sequences.
  - Inputs: tree sequence file(s) from SLiM simulation and a YAML configuration file specifying the demographic model to infer
  - Outputs: Synonymous and nonsynonymous SFS files (saved as dadi.Spectrum objects) and optimization results (.txt) file
  - Example command line usage at top of script
- auto_demographic_functions.py: Additional dadi demographic model functions for autotetraploids
- allo_demographic_functions.py: Additional dadi demographic model functions for allotetraploids
- auto_demog_config.yaml: Example YAML configuration file for demographic inference for autotetraploid
- allo_demog_config.yaml: Example YAML configuration file for demographic inference for allotetraploid

### DFE_inference_and_caches
This folder contains scripts and additional files for constructing caches and inferring DFE models.

The files are:

- dadi_cache_generation.py: Script for generating 1D and 2D caches for dadi DFE inference.
  - Inputs: nonsynonymous SFS file (from dadi_demographic_inference.py) and demographic optimization results file from demographic inference (from dadi_demographic_inference.py)
  - Outputs: 1D and 2D cache files (saved as pickle files)
  - Example command line usage at top of script
- dadi_DFE_inference.py: Script for inferring DFE models from nonsynonymous SFS and optimization results files.
  - Inputs: nonsynonymous SFS file (from dadi_demographic_inference.py), demographic optimization results file (from dadi_demographic_inference.py), caches (from dadi_cache_generation.py), and a YAML configuration file specifying the DFE model to infer
  - Outputs: DFE optimization results file (saved as .txt file) and a pdf of the resulting DFE fit to the nonsynonymous SFS
  - Example command line usage at top of script
- allo_demographic_functions_sel.py: Additional dadi demographic model functions for allotetraploids with selection
- gamma_DFE_config.yaml: Example YAML configuration file for DFE inference for gamma DFE with a 1D cache
- lognormal_joint_DFE_config.yaml: Example YAML configuration file for DFE inference for lognormal mixture DFE with a 2D cache
