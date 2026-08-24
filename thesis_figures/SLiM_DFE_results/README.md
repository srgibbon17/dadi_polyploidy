# SLiM_DFE_results
Code and data for validation of DFE inference with dadi and fastDFE based on SLiM simulations.

### Subdirectories and figures
- `Allotetraploids`
  - Demographic inference and dadi DFE inference for an allotetraploid two-size change model
  - `demography_box_plots.pdf`: Box plots for demographic inference parameters
  - `dfe_scatter_small.pdf` and `dfe_scatter_large.pdf`: Scatter plots of inferred DFE shape and scale parameters for small (~20) and large (~100) sample sizes, respectively
  - `dfe_discretized_small.pdf` and `dfe_discretized_large.pdf`: Discretized gamma DFE fits compared to the true simulated discretized gamma DFE. For each sample size, the gamma DFE is discretized for each replicate and then the discrete bins are averaged over replicates to show a single discretized DFE.
- `Bottlegrowth`
  - Demographic inference, dadi and fastDFE DFE inference for an autotetraploid bottlegrowth model
  - `demography_box_plots.pdf`: Box plots for demographic inference parameters
  - `dfe_scatter_small.pdf`, `dfe_scatter_med.pdf`, and `dfe_scatter_large.pdf`: Scatter plots of inferred DFE shape and scale parameters for small (~20), medium (~50), and large (~100) sample sizes, respectively
  - `dfe_discretized_small.pdf`, `dfe_discretized_med.pdf`, and `dfe_discretized_large.pdf`: Discretized gamma DFE fits compared to the true simulated discretized gamma DFE. For each sample size, the gamma DFE is discretized for each replicate and then the discrete bins are averaged over replicates to show a single discretized DFE.
- `Joint_DFE`
  - Demographic inference and dadi DFE infernce for an autotetraploid and diploid progenitor joint DFE model
  - `demography_box_plots.pdf`: Box plots for demographic inference parameters
  - `dfe_scatter_small.pdf` and `dfe_scatter_large.pdf`: Scatter plots of inferred DFE shape and scale parameters for small (~20) and large (~100) sample sizes, respectively
  - `dfe_discretized_small.pdf` and `dfe_discretized_large.pdf`: Discretized gamma DFE fits compared to the true simulated discretized gamma DFE. For each sample size, the gamma DFE is discretized for each replicate and then the discrete bins are averaged over replicates to show a single discretized DFE.
  - `dfe_w_boxplot.pdf`: Box plot of the joint DFE mixture model correlation parameter w
- `Joint_DFE_migration`
  - Demographic inference and dadi DFE infernce for an autotetraploid and diploid progenitor joint DFE model
  - `demography_box_plots.pdf`: Box plots for demographic inference parameters
  - `dfe_scatter_large.pdf`: Scatter plots of inferred DFE shape and scale parameters for large (~100) sample sizes, respectively
  - `dfe_discretized_large.pdf`: Discretized gamma DFE fits compared to the true simulated discretized gamma DFE. For each sample size, the gamma DFE is discretized for each replicate and then the discrete bins are averaged over replicates to show a single discretized DFE.
  - `dfe_w_boxplot.pdf`: Box plot of the joint DFE mixture model correlation parameter w