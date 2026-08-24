"""
Generate SFS samples from msprime for inference with dadi.
This allows us to test dadi's sensitivity to differentiating between
auto, allo, and allo w/ HE models. 

To test this, we can use a combination of AIC/BIC 
(for the case of non-nested models) or LRT (for nested models).

This also allows us to test the ability of dadi to infer 
demographic parameters from data with different ploidy types.

Command line usage:
python3 path_to_script/generate_msprime_samples.py \
    --config-file path_to_config_file/demography_configs_ex.yaml 

See the demography_configs_ex.yaml file for an example of the config file.
"""

import msprime
import numpy as np
import os
import yaml
import argparse

# also note that, for simplicity, for the allotetraploid models, we model divergence between 
# the diploid progenitors of each subgenome, but fix this to be for a time of 2*Na generations (i.e. 1 diffusion unit)

# we hard code the following parameters for the demographic models in msprime
# L: genome length (in base pairs) = 1e7
# r: recombination rate = 1e-8
# mu: mutation rate = 1e-8
# Na: ancestral (diploid) population size = 10000 (1e4)

def auto_2epoch(nu_c, T_WGD, output_dir, seed=42, replicates=1, ns=(40,)):
    """
    Simulate a simple demographic model of allotetraploid formation using msprime. 
    Save the output (one for each replicate) to a .trees file.

    nu_c: size of the current population of autotetraploids relative to ancestral population size
    T_WGD: time in the past at which the WGD occurred, creating the  
            autotetraploid population (in units of 2*Na*generations)
    output_dir: directory to save the output files
    seed: a random seed for reproducibility
    num_samples: number of samples (of 10 tetraploid individuals)
            to draw from each simulation
    replicates: number of separate simulations to run
    ns: sample sizes (in number of haplotypes) for each "population" (or subgenome)
        this matches the dadi convention... here, sampling 4 haplotypes is equivalent to sampling 1 autotetraploid
    """
    if len(ns) != 1:
        raise ValueError("ns must be a tuple of length 1")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # set up the msprime demography
    Na = 10000
    demography = msprime.Demography()
    demography.add_population(initial_size=Na, name="ancestral")
    demography.add_population(initial_size=int(2*nu_c*Na), name="autos") # set the size of the current population of autos
    demography.add_population_split(time=int(T_WGD*2*Na), derived=["autos"], ancestral="ancestral") # set up the WGD event
    
    # set up a function to run the simulation
    def run_sim(demography, samples_list, anc_seed, mut_seed):
        ts = msprime.sim_ancestry(samples=samples_list,  
                                sequence_length=1e7, recombination_rate=1e-8, 
                                ploidy=2, demography=demography, random_seed=anc_seed)
        mutated_ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=mut_seed)
        return mutated_ts

    # set up the sample sets used by msprime
    samples_list = []
    samples_list.append(msprime.SampleSet(ns[0], "autos", ploidy=1))

    # set up seeds
    rng = np.random.RandomState(seed)
    seeds = rng.randint(1, 2**31, size=(replicates, 2))

    for rep_index in range(replicates):
        # run the simulation
        ts = run_sim(demography, samples_list, *seeds[rep_index])
        
        # save the ts to a .trees file (which includes the provenance information)
        ts.dump(os.path.join(output_dir, f"auto_2epoch_T{T_WGD}_nu{nu_c}_master{seed}_rep{rep_index}.trees"))

def allo_2epoch(nu_c, T_WGD, HE_rate, output_dir, seed=42, replicates=1, ns=(20,20)):
    """
    Simulate a simple demographic model of allotetraploid formation using msprime. 
    Save the output (one for each replicate) to a .trees file.

    nu_c: size of the current population of allotetraploids relative to ancestral population size
    T_WGD: time in the past at which the WGD occurred, creating the  
           allotetraploid population (in units of 2*Na*generations)
    HE_rate: population-scaled rate of homoeologous exchanges between subgenomes
    output_dir: directory to save the output files
    seed: a random seed for reproducibility
    replicates: number of replicate simulations to run
    ns: sample sizes (in number of haplotypes) for each "population" (or subgenome)
        this matches the dadi convention... here, sampling 2 haplotypes is equivalent to sampling one of the subgenomes
    """
    if len(ns) != 2:
        raise ValueError("ns must be a tuple of length 2")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    Na=10000
    demography = msprime.Demography()
    demography.add_population(initial_size=Na, name="ancestral")
    demography.add_population(initial_size=Na, name="dipa")
    demography.add_population(initial_size=Na, name="dipb")
    demography.add_population(initial_size=int(nu_c*Na), name="allosa") # set the size of the current population of allos
    demography.add_population(initial_size=int(nu_c*Na), name="allosb") # set the size of the current population of allos
    demography.add_population_split(time=int(T_WGD*2*Na), derived=["allosa"], ancestral="dipa") # set up the WGD event
    demography.add_population_split(time=int(T_WGD*2*Na), derived=["allosb"], ancestral="dipb") # set up the WGD event
    demography.add_population_split(time=int(2*Na*(1+T_WGD)), derived=["dipa", "dipb"], ancestral="ancestral") # set up the divergence event
    demography.set_symmetric_migration_rate(populations=['allosa', 'allosb'], rate=HE_rate/(2*Na))

    # set up a function to run the simulation
    def run_sim(demography, samples_list, anc_seed, mut_seed):
        ts = msprime.sim_ancestry(samples=samples_list, sequence_length=1e7, recombination_rate=1e-8, 
                                ploidy=2, demography=demography, random_seed=anc_seed)
        mutated_ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=mut_seed)
        return mutated_ts
    
    # set up the sample sets used by msprime
    samples_list = []
    samples_list.append(msprime.SampleSet(ns[0], population="allosa", ploidy=1))
    samples_list.append(msprime.SampleSet(ns[1], population="allosb", ploidy=1))
    
    # set up seeds for ancestry and mutation simulations
    rng = np.random.RandomState(seed)
    seeds = rng.randint(1, 2**31, size=(replicates, 2))

    for rep_index in range(replicates):
        # run the simulation
        ts = run_sim(demography, samples_list, *seeds[rep_index])

        # save the ts to a .trees file (which includes the provenance information)
        ts.dump(os.path.join(output_dir, f"allo_2epoch_T{T_WGD}_nu{nu_c}_H{HE_rate}_master{seed}_rep{rep_index}.trees"))

def auto_bottlegrowth(nu_WGD, nu_c, T_WGD, output_dir, seed=42, replicates=1, ns=(40,)):
    """
    Set up a bottlegrowth demography for msprime modeling autotetraploid formation.

    nu_WGD: size of the population of autotetraploids immediately after
            the WGD event relative to ancestral population size
    nu_c: size of the current population of autotetraploids relative to ancestral population size
    T_WGD: time in the past at which the WGD occurred, creating the  
           autotetraploid population (in units of 2*Na*generations)
    output_dir: directory to save the output files
    seed: a random seed for reproducibility
    replicates: number of separate simulations to run
    ns: sample sizes (in number of haplotypes) for each "population" (or subgenome)
        this matches the dadi convention... here, sampling 4 haplotypes is equivalent to sampling 1 autotetraploid
    """
    if len(ns) != 1:
        raise ValueError("ns must be a tuple of length 1")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # set up the msprime demography
    Na = 10000
    alpha = -np.log(nu_WGD/nu_c)/(2*Na*T_WGD) # solve for the growth rate alpha in NCe^(-alpha*T_WGD) = NWGD for msprime units
    demography = msprime.Demography()
    demography.add_population(initial_size=10000, name="ancestral")
    demography.add_population(initial_size=int(2*nu_c*Na), name="autos", growth_rate=alpha) # set the size of the first epoch population of autos
    demography.add_population_split(time=int(2*Na*T_WGD), derived=["autos"], ancestral="ancestral") # set up the WGD event

    # set up a function to run the simulation
    def run_sim(demography, samples_list, anc_seed, mut_seed):
        ts = msprime.sim_ancestry(samples=samples_list,  
                                sequence_length=1e7, recombination_rate=1e-8, 
                                ploidy=2, demography=demography, random_seed=anc_seed)
        mutated_ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=mut_seed)
        return mutated_ts

    # set up the sample sets used by msprime
    samples_list = []
    samples_list.append(msprime.SampleSet(ns[0], "autos", ploidy=1))

    # set up seeds
    rng = np.random.RandomState(seed)
    seeds = rng.randint(1, 2**31, size=(replicates, 2))

    for rep_index in range(replicates):
        # run a simulation with 10 samples of size 40 each
        ts = run_sim(demography, samples_list, *seeds[rep_index])
        
        ts.dump(os.path.join(output_dir, f"auto_bottlegrowth_T{T_WGD}_nu{nu_WGD}_nuc{nu_c}_master{seed}_rep{rep_index}.trees"))

def allo_bottlegrowth(nu_WGD, nu_c, T_WGD, HE_rate, output_dir, seed=42, replicates=1, ns=(20,20)):
    """
    Set up a two epoch demography for msprime modeling allotetraploid formation.

    nu_WGD: size of the population of allotetraploids immediately after
            the WGD event relative to the ancestral population size
    nu_c: size of the current population of allotetraploids relative to the ancestral population size
    T_WGD: time in the past at which the WGD occurred, creating the  
           allotetraploid population (in units of 2*Na*generations)
    HE_rate: population-scaled rate of homoeologous exchanges between subgenomes 
    output_dir: directory to save the output files
    seed: a random seed for reproducibility
    num_samples: number of samples (of 10 tetraploid individuals)
            to draw from each simulation
    replicates: number of separate simulations to run
    ns: sample sizes (in number of haplotypes) for each "population" (or subgenome)
        this matches the dadi convention... here, sampling 2 haplotypes is equivalent to sampling one of the subgenomes
    """
    if len(ns) != 2:
        raise ValueError("ns must be a tuple of length 2")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # set up the msprime demography
    Na=10000
    alpha = -np.log(nu_WGD/nu_c)/(2*Na*T_WGD) # solve for the growth rate alpha in NCe^(-alpha*T_WGD) = NWGD for msprime units
    demography = msprime.Demography()
    demography.add_population(initial_size=Na, name="ancestral")
    demography.add_population(initial_size=Na, name="dipa")
    demography.add_population(initial_size=Na, name="dipb")
    demography.add_population(initial_size=int(nu_c*Na), name="allosa", growth_rate=alpha) # set the size of the current population of allos
    demography.add_population(initial_size=int(nu_c*Na), name="allosb", growth_rate=alpha) # set the size of the current population of allos
    demography.add_population_split(time=int(T_WGD*2*Na), derived=["allosa"], ancestral="dipa") # set up the WGD event
    demography.add_population_split(time=int(2*Na*T_WGD), derived=["allosb"], ancestral="dipb") # set up the WGD event
    demography.add_population_split(time=int(2*Na*(1+T_WGD)), derived=["dipa", "dipb"], ancestral="ancestral") # set up the divergence event
    demography.set_symmetric_migration_rate(populations=['allosa', 'allosb'], rate=HE_rate/(2*Na))

    # set up a function to run the simulation
    def run_sim(demography, samples_list, anc_seed, mut_seed):
        ts = msprime.sim_ancestry(samples=samples_list,  
                                sequence_length=1e7, recombination_rate=1e-8, 
                                ploidy=2, demography=demography, random_seed=anc_seed)
        mutated_ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=mut_seed)
        return mutated_ts
    
    # set up the sample sets used by msprime
    samples_list = []
    samples_list.append(msprime.SampleSet(ns[0], "allosa", ploidy=1))
    samples_list.append(msprime.SampleSet(ns[1], "allosb", ploidy=1))

    # set up seeds
    rng = np.random.RandomState(seed)
    seeds = rng.randint(1, 2**31, size=(replicates, 2))

    for rep_index in range(replicates):
        # run the simulation
        ts = run_sim(demography, samples_list, *seeds[rep_index])
        
        # save the ts to a .trees file (which includes the provenance information)
        ts.dump(os.path.join(output_dir, f"allo_bottlegrowth_T{T_WGD:.4f}_nuWGD{nu_WGD:.4f}_nuc{nu_c:.4f}_H{HE_rate:.4f}_master{seed}_rep{rep_index}.trees"))

def auto_triple_recurrent(T_WGD1, T_WGD2, nu_T1_WGD, nu_T1_c, nu_T2_WGD, nu_T2_c, 
                          nu_D_WGD1, nu_D_WGD2, M_tets, M_T1_D, M_T2_D, 
                          output_dir, seed=42, replicates=1, ns=(18,24,24)):
    """
    Set up a demographic model of recurrent formation of two autotetraploid populations 
    from a single ancestral diploid population (i.e., each autotetraploid pop splits from the diploid).
    We name the populations "D" for diploid, "T1" for the first autotetraploid,
    and "T2" for the second autotetraploid. So, T1 splits from D and then T2 later splits from D. 
    The diploid population has a two epoch model (with size changes pinned to the WGD event times).
    Each autoetraploid population has a bottlegrowth model. 

    Parameters:
    -----------
    Note: *all* parameters are in diffusion units (i.e., time in 2*Na*generations, M)
    T_WGD1: time in the past at which the first WGD occurred, creating the  
           first autotetraploid population (in units of 2*Na*generations)
    T_WGD2: time in the past at which the second WGD occurred, creating the  
           second autotetraploid population (in units of 2*Na*generations)
    nu_T1_WGD: relative size of the first bottlenecked auotetraploid population
    nu_T1_c: relative size of the first contemporary autotetraploid population
    nu_T2_WGD: relative size of the second bottlenecked auotetraploid population
    nu_T2_c: relative size of the second contemporary autotetraploid population
    nu_D_WGD1: relative size of the diploid population between the two WGD events
    nu_D_WGD2: relative size of the diploid population after the second WGD event
    M_tets: symmetric migration rate from the first autotetraploid to the second autotetraploid
    M_T1_D: symmetric migration rate from the the diploids to the first autotetraploids
    M_T2_D: symmetric migration rate from the the diploids to the second autotetraploids

    output_dir: directory to save the output files
    seed: a random seed for reproducibility
    replicates: number of separate simulations to run
    ns: sample sizes = (ns_D, ns_T1, ns_T2) (in number of haplotypes) for each "population" (or subgenome)
        this matches the dadi convention... here, sampling 4 haplotypes is equivalent to sampling 1 autotetraploid
    """
    if len(ns) != 3:
        raise ValueError("ns must be a tuple of length 3")
    if T_WGD1 < T_WGD2:
        raise ValueError("T_WGD1 must be greater than T_WGD2")
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # set up the msprime demography
    Na = 10000
    alpha_T1 = -np.log(nu_T1_WGD/nu_T1_c)/(2*Na*T_WGD1) # solve for the growth rate alpha in NCe^(-alpha*T_WGD) = NWGD for msprime units
    alpha_T2 = -np.log(nu_T2_WGD/nu_T2_c)/(2*Na*T_WGD2) # solve for the growth rate alpha in NCe^(-alpha*T_WGD) = NWGD for msprime units
    demography = msprime.Demography()
    # add the ancestral diploid population
    demography.add_population(initial_size=10000, name="ancestral")
    # then add the intermediate diploid population and contemporary diploid population
    demography.add_population(initial_size=int(nu_D_WGD1*Na), name="D_intermediate") 
    demography.add_population(initial_size=int(nu_D_WGD2*Na), name="D")
    # then add an intermediate population for T1
    # note: here, the initial size is the size at T=0 (i.e., present day), so we just set it to 2*nu_T1_c*Na
    demography.add_population(initial_size=int(2*nu_T1_c*Na), name="T1_intermediate", growth_rate=alpha_T1)
    # then add the two contemporary autotetraploid populations 
    demography.add_population(initial_size=int(2*nu_T1_c*Na), name="T1", growth_rate=alpha_T1) 
    demography.add_population(initial_size=int(2*nu_T2_c*Na), name="T2", growth_rate=alpha_T2)
    # then add the first WGD
    demography.add_population_split(time=int(2*Na*T_WGD1), derived=["D_intermediate", "T1_intermediate"], ancestral="ancestral") 
    # and the second WGD
    demography.add_population_split(time=int(2*Na*T_WGD2), derived=["T1"], ancestral="T1_intermediate")
    demography.add_population_split(time=int(2*Na*T_WGD2), derived=["D", "T2"], ancestral="D_intermediate") 
    # then add the migration rates
    demography.set_migration_rate(source="T1_intermediate", dest="D_intermediate", rate=M_T1_D/(2*Na))
    demography.set_symmetric_migration_rate(populations=['T1', 'T2'], rate=M_tets/(2*Na))
    demography.set_migration_rate(source="T1", dest="D", rate=M_T1_D/(2*Na))
    demography.set_migration_rate(source="T2", dest="D", rate=M_T2_D/(2*Na))

    demography.sort_events()

    # set up a function to run the simulation
    def run_sim(demography, samples_list, anc_seed, mut_seed):
        ts = msprime.sim_ancestry(samples=samples_list,  
                                sequence_length=1e7, recombination_rate=1e-8, 
                                ploidy=2, demography=demography, random_seed=anc_seed)
        mutated_ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=mut_seed)
        return mutated_ts

    # set up the sample sets used by msprime
    samples_list = []
    samples_list.append(msprime.SampleSet(ns[0], "D", ploidy=1)) # Diploid samples
    samples_list.append(msprime.SampleSet(ns[1], "T1", ploidy=1)) # First autotetraploid samples
    samples_list.append(msprime.SampleSet(ns[2], "T2", ploidy=1)) # Second autotetraploid samples

    # set up seeds
    rng = np.random.RandomState(seed)
    seeds = rng.randint(1, 2**31, size=(replicates, 2))

    for rep_index in range(replicates):
        # run a simulation with 10 samples of size 40 each
        ts = run_sim(demography, samples_list, *seeds[rep_index])
        
        ts.dump(os.path.join(output_dir, f"auto_triple_recurrent_master{seed}_rep{rep_index}.trees"))

def auto_triple_single_origin(T_WGD1, T_WGD2, nu_T1_WGD, nu_T1_c, nu_T2_WGD, nu_T2_c, 
                          nu_D_WGD1, nu_D_WGD2, M_tets, M_T1_D, M_T2_D, 
                          output_dir, seed=42, replicates=1, ns=(18,24,24)):
    """
    Set up a demographic model of two autotetraploid populations with a single origin
    from one ancestral diploid population (i.e., each autotetraploid pop splits from the diploid).
    We name the populations "D" for diploid, "T1" for the first autotetraploid,
    and "T2" for the second autotetraploid. So, T1 splits from D and then T2 later splits from T1.
    The diploid population has a two epoch model (with size changes pinned to the WGD event times).
    Each autoetraploid population has a bottlegrowth model. 

    Parameters:
    -----------
    Note: *all* parameters are in diffusion units (i.e., time in 2*Na*generations, M)
    T_WGD1: time in the past at which the first WGD occurred, creating the  
           first autotetraploid population (in units of 2*Na*generations)
    T_WGD2: time in the past at which the second WGD occurred, creating the  
           second autotetraploid population (in units of 2*Na*generations)
    nu_T1_WGD: relative size of the first bottlenecked auotetraploid population
    nu_T1_c: relative size of the first contemporary autotetraploid population
    nu_T2_WGD: relative size of the second bottlenecked auotetraploid population
    nu_T2_c: relative size of the second contemporary autotetraploid population
    nu_D_WGD1: relative size of the diploid population between the two WGD events
    nu_D_WGD2: relative size of the diploid population after the second WGD event
    M_tets: symmetric migration rate from the first autotetraploid to the second autotetraploid
    M_T1_D: symmetric migration rate from the the diploids to the first autotetraploids
    M_T2_D: symmetric migration rate from the the diploids to the second autotetraploids

    output_dir: directory to save the output files
    seed: a random seed for reproducibility
    replicates: number of separate simulations to run
    ns: sample sizes = (ns_D, ns_T1, ns_T2) (in number of haplotypes) for each "population" (or subgenome)
        this matches the dadi convention... here, sampling 4 haplotypes is equivalent to sampling 1 autotetraploid
    """
    if len(ns) != 3:
        raise ValueError("ns must be a tuple of length 3")
    if T_WGD1 < T_WGD2:
        raise ValueError("T_WGD1 must be greater than T_WGD2")
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # set up the msprime demography
    Na = 10000
    alpha_T1 = -np.log(nu_T1_WGD/nu_T1_c)/(2*Na*T_WGD1) # solve for the growth rate alpha in NCe^(-alpha*T_WGD) = NWGD for msprime units
    alpha_T2 = -np.log(nu_T2_WGD/nu_T2_c)/(2*Na*T_WGD2) # solve for the growth rate alpha in NCe^(-alpha*T_WGD) = NWGD for msprime units
    demography = msprime.Demography()
    # add the ancestral diploid population
    demography.add_population(initial_size=10000, name="ancestral")
    # then add the intermediate diploid population and contemporary diploid population
    demography.add_population(initial_size=int(nu_D_WGD1*Na), name="D_intermediate") 
    demography.add_population(initial_size=int(nu_D_WGD2*Na), name="D")
    # then add an intermediate population for T1
    # note: here, the initial size is the size at T=0 (i.e., present day), so we just set it to 2*nu_T1_c*Na
    demography.add_population(initial_size=int(2*nu_T1_c*Na), name="T1_intermediate", growth_rate=alpha_T1)
    # then add the two contemporary autotetraploid populations 
    demography.add_population(initial_size=int(2*nu_T1_c*Na), name="T1", growth_rate=alpha_T1) 
    demography.add_population(initial_size=int(2*nu_T2_c*Na), name="T2", growth_rate=alpha_T2)
    # then add the first WGD
    demography.add_population_split(time=int(2*Na*T_WGD1), derived=["D_intermediate", "T1_intermediate"], ancestral="ancestral") 
    # and the second WGD - this is the only difference from the recurrent formation model
    demography.add_population_split(time=int(2*Na*T_WGD2), derived=["T1", "T2"], ancestral="T1_intermediate")
    demography.add_population_split(time=int(2*Na*T_WGD2), derived=["D"], ancestral="D_intermediate") 
    # then add the migration rates
    demography.set_migration_rate(source="T1_intermediate", dest="D_intermediate", rate=M_T1_D/(2*Na))
    demography.set_symmetric_migration_rate(populations=['T1', 'T2'], rate=M_tets/(2*Na))
    demography.set_migration_rate(source="T1", dest="D", rate=M_T1_D/(2*Na))
    demography.set_migration_rate(source="T2", dest="D", rate=M_T2_D/(2*Na))

    demography.sort_events()

    # set up a function to run the simulation
    def run_sim(demography, samples_list, anc_seed, mut_seed):
        ts = msprime.sim_ancestry(samples=samples_list,  
                                sequence_length=1e7, recombination_rate=1e-8, 
                                ploidy=2, demography=demography, random_seed=anc_seed)
        mutated_ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=mut_seed)
        return mutated_ts

    # set up the sample sets used by msprime
    samples_list = []
    samples_list.append(msprime.SampleSet(ns[0], "D", ploidy=1)) # Diploid samples
    samples_list.append(msprime.SampleSet(ns[1], "T1", ploidy=1)) # First autotetraploid samples
    samples_list.append(msprime.SampleSet(ns[2], "T2", ploidy=1)) # Second autotetraploid samples

    # set up seeds
    rng = np.random.RandomState(seed)
    seeds = rng.randint(1, 2**31, size=(replicates, 2))

    for rep_index in range(replicates):
        # run a simulation with 10 samples of size 40 each
        ts = run_sim(demography, samples_list, *seeds[rep_index])
        
        ts.dump(os.path.join(output_dir, f"auto_triple_recurrent_master{seed}_rep{rep_index}.trees"))


def get_model_function(model_name):
    """
    Map model name string to actual function object.    
    """
    model_map = {
        'allo_2epoch': allo_2epoch,
        'auto_2epoch': auto_2epoch,
        'allo_bottlegrowth': allo_bottlegrowth,
        'auto_bottlegrowth': auto_bottlegrowth,
        'auto_triple_recurrent': auto_triple_recurrent,
        'auto_triple_single_origin': auto_triple_single_origin
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown model function: {model_name}")    
    
    return model_map[model_name]

def main():
    parser = argparse.ArgumentParser(
        description='Run demographic models for msprime simulations.'
    )
    parser.add_argument(
        '--config-file',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config_data = yaml.safe_load(f)

    base_dir = config_data['base_dir']
    configs = config_data['configs']
    model = config_data['model']

    # Get model function
    try:
        model_func = get_model_function(model)
    except ValueError as e:
        print(f"ERROR: {e}")
        
    # run the simulations
    for config in configs:
        try:
            params = config['params']
            output_dir = base_dir + config['output_dir']
            seed = config['seed']
            replicates = config['replicates']
            ns = config['ns']
            model_func(*params, output_dir, seed=seed, replicates=replicates, ns=ns)

        except Exception as e:
            print(f"ERROR: Failed to run simulation: {e}")

if __name__ == '__main__':
    main()




### Some older scripts for three epoch models, which I don't think I am going to use
# def auto_3epoch(Na, NWGD, Nc, T_WGD, TF):
#     """
#     Set up a three epoch demography for msprime modeling autotetraploid formation.

#     Na: ancestral population size (number of individuals)
#     NWGD: size of the population of autotetraploids immediately after
#             the WGD event (in effective diploid size)
#     NC: size of the current population of autotetraploids (in effective diploid size)
#     T_WGD: time in the past at which the WGD occurred, creating the  
#            autotetraploid population (in units of generations)
#     TF: time in the past at which the second epoch began (in units of generations)
#     """
#     demography = msprime.Demography()
#     demography.add_population(initial_size=Na, name="ancestral")
#     demography.add_population(initial_size=NWGD, name="autos1") # set the size of the first epoch population of autos
#     demography.add_population(initial_size=Nc, name="autos2") # set the size of the current population of autos
#     demography.add_population_split(time=TF, derived=["autos2"], ancestral="autos1") # set up the second epoch
#     demography.add_population_split(time=T_WGD, derived=["autos1"], ancestral="ancestral") # set up the WGD event
#     return demography

# def allo_3epoch(Na, NWGD, Nc, T_WGD, TF, HE_rate):
#     """
#     Set up a three epoch demography for msprime modeling allotetraploid formation.

#     Na: ancestral population size (number of individuals)
#     NWGD: size of the population of allotetraploids immediately after
#             the WGD event (in effective diploid size)
#     Nc: size of the current population of allotetraploids (in effective diploid size)
#     T_WGD: time in the past at which the WGD occurred, creating the  
#            allotetraploid population (in units of generations)
#     TF: time in the past at which the second epoch began (in units of generations)
#     HE_rate: rate of homoeologous exchanges between subgenomes (this is constant over both epochs)
#     """
#     demography = msprime.Demography()
#     demography.add_population(initial_size=Na, name="ancestral")
#     demography.add_population(initial_size=Na, name="dipa")
#     demography.add_population(initial_size=Na, name="dipb")
#     demography.add_population(initial_size=NWGD, name="allosa1") # set the size of the first epoch population of allos
#     demography.add_population(initial_size=NWGD, name="allosb1") # set the size of the first epoch population of allos
#     demography.add_population(initial_size=Nc, name="allosa2") # set the size of the current population of allos
#     demography.add_population(initial_size=Nc, name="allosb2") # set the size of the current population of allos
#     demography.add_population_split(time=TF, derived=["allosa2"], ancestral="allosa1") # set up the WGD event
#     demography.add_population_split(time=TF, derived=["allosb2"], ancestral="allosb1") # set up the WGD event
#     demography.add_population_split(time=T_WGD, derived=["allosa1"], ancestral="dipa") # set up the WGD event
#     demography.add_population_split(time=T_WGD, derived=["allosb1"], ancestral="dipb") # set up the WGD event
#     demography.add_population_split(time=2*Na+T_WGD, derived=["dipa", "dipb"], ancestral="ancestral") # set up the divergence event
#     demography.set_symmetric_migration_rate(populations=['allosa1', 'allosb1'], rate=HE_rate)
#     demography.set_symmetric_migration_rate(populations=['allosa2', 'allosb2'], rate=HE_rate)
#     return demography