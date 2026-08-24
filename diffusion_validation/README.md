# diffusion_validation
A set of Jupyter notebooks for validating the numerical integration of the diffusion equation for polyploids as implemented in `dadi.Polyploidy.Integration`.

### Integration checks and unit tests
The `integration_unit_tests` directory contains a set of notebooks (one for each integration dimension) which compares the numerical integration in `dadi.Polyploidy` vs rescaled integration using base `dadi` or Wright-Fisher simulations. It also contains a notebook which checks all of the unit tests for the polyploidy integration module of dadi.

### Wright-Fisher simulations
The `WF_comparisons` directory contains a set of notebooks which make visual comparisons between the numerical integration in `dadi.Polyploidy` and custom Wright-Fisher simulations implemented with `numpy`. 
