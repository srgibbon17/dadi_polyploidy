# autotetraploid_topologies
Results and figures from msprime simulations of autotetraploid populations under two competing demographic histories: a single origin model (where an autotetraploid population forms once and splits later) and a recurrent formation model (where the autotetraploid populations forms twice from the same diploid progenitor population). `auto_triplet.ipynb` contains all the code to reproduce the figures listed below.

### Figures
- `autotet_formation.pdf` Main figure for my thesis showing the competing demographic models, the inferred parameters from the single origin simulations under both models, and the AIC, BIC, CLAIC, and CLBIC values for pairs of inferences. `single_comparison_v2.pdf` plots all the data without the demographic models (which are added in post with Adobe). 
- `recurrent_formation.pdf` Figure showing parameter estimates and AIC, BIC, etc. for simulations under the recurrent formation model.
- `single_recurrent_demes.pdf` Figure showing the demesdraw demographic model for the single origin simulations under the recurrent formation model.
- `effective_dimensions.pdf` Figure showing estimates of the effective number of parameters in the demographic model which is used in calculating the CLAIC and CLBIC statistics.

### Data
Summaries of the AIC, BIC, and best fit parameters produced from the pipeline described in `dadi_polyploidy/msprime_demography_validation` are available for both the single origin and recurrent formation models. 
