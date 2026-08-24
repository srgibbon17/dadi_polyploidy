### The below is a set of slightly modified demographic functions from 
### allo_demographics.py in the dadi distribution.
### Most notably, we fix T_div = 1 in the below models. 

from dadi import Numerics, PhiManip
from dadi.Spectrum_mod import Spectrum
from dadi.Polyploidy import Integration as PolyInt
import numpy

def two_epoch_sel(params, ns, pts):
    """
    Two epoch model of allotetraploid formation where the 
    diploid progenitors diverge for 1 diffusion unit and then the
    allotetraploid population splits and maintains a size of nu.
    
    Parameters:
        params (tuple): (T_WGD, nu, H, gamma)
            - T_WGD: Time in the past at which the WGD occurred, creating the  
               autotetraploid population (in units of 2*Na generations).

            - nu: Ratio of contemporary autotetraploid to ancient diploid population size 
               (ratio of *census* sizes).

            - H: homoeologous exchange rate (in terms of 2*Na*eta)

            - gamma: population-scaled selection coefficient (= 2*Na*s)
        ns (tuple): Sample sizes (n1,n2).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting (collapsed) frequency spectrum.
    """
    T_WGD, nu, H, gamma = params
    alloaflag = PolyInt.PloidyType.ALLOa
    allobflag = PolyInt.PloidyType.ALLOb
    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx, gamma=gamma)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    # integrate for T=1 to model diploid divergence
    phi = PolyInt.two_pops(phi, xx, 1, sel_dict1={"gamma": gamma}, sel_dict2={"gamma": gamma})
    # then, integrate for T_WGD with allotetraploids
    phi = PolyInt.two_pops(phi, xx, T_WGD, nu1=nu, nu2=nu, m12=H, m21=H,
                           sel_dict1={"gamma": gamma}, sel_dict2={"gamma": gamma},
                           ploidyflag1=alloaflag, ploidyflag2=allobflag)
    fs = Spectrum.from_phi(phi, ns, (xx,xx))
    return fs
two_epoch_sel.__param_names__ = ['T_WGD', 'nu', 'H', 'gamma']

def two_epoch_noHE_sel(params, ns, pts):
    """
    Two epoch model of allotetraploid formation where the 
    diploid progenitors diverge for 1 diffusion unit and then the
    allotetraploid population splits and maintains a size of nu.
    
    Parameters:
        params (tuple): (T_WGD, nu, gamma)
            - T_WGD: Time in the past at which the WGD occurred, creating the  
               autotetraploid population (in units of 2*Na generations).

            - nu: Ratio of contemporary autotetraploid to ancient diploid population size 
               (ratio of *census* sizes).

            - gamma: population-scaled selection coefficient (= 2*Na*s)
        ns (tuple): Sample sizes (n1,n2).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting (collapsed) frequency spectrum.
    """
    T_WGD, nu, gamma = params
    fs = two_epoch_sel((T_WGD, nu, 0, gamma), ns, pts)
    return fs
two_epoch_noHE_sel.__param_names__ = ['T_WGD', 'nu', 'gamma']


def bottlegrowth_sel(params, ns, pts):
    """
    Bottlegrowth model of allotetraploid formation where the 
    allotetraploid population starts with size nuWGD and 
    grows exponentially to a size of nuF
    
    Parameters:
        params (tuple): (T_WGD, nuWGD, nuF, H, gamma)

            - T_WGD: Time in the past at which the WGD occurred, creating the  
               allotetraploid population (in units of 2*Na generations).

            - nuWGD: Ratio of allotetraploid population immediately after WGD
                to ancient diploid population size (ratio of *census* sizes).

            - nuF: Ratio of contemporary allotetraploid population
                to ancient diploid population size (ratio of *census* sizes).

            - H: homoeologous exchange rate (in terms of 2*Na*eta)

            - gamma: population-scaled selection coefficient (= 2*Na*s)
        ns (tuple): Sample sizes (n1,n2).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    T_WGD, nuWGD, nuF, H, gamma = params
    nu_f = lambda t: nuWGD*numpy.exp(numpy.log(nuF/nuWGD) * t/T_WGD)
    alloaflag = PolyInt.PloidyType.ALLOa
    allobflag = PolyInt.PloidyType.ALLOb
    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx, gamma=gamma)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    # T=1 divergence period between diploids
    phi = PolyInt.two_pops(phi, xx, 1, sel_dict1={"gamma": gamma}, sel_dict2={"gamma": gamma})
    phi = PolyInt.two_pops(phi, xx, T_WGD, nu=nu_f, m12=H, m21=H,
                           sel_dict1={"gamma": gamma}, sel_dict2={"gamma": gamma},
                           ploidyflag1=alloaflag, ploidyflag2=allobflag)
    fs = Spectrum.from_phi(phi, ns, (xx,xx))
    return fs
bottlegrowth_sel.__param_names__ = ["T_WGD", "nuWGD", "nuF", "H", "gamma"]

def bottlegrowth_noHE_sel(params, ns, pts):
    """
    Bottlegrowth model of allotetraploid formation where the 
    allotetraploid population starts with size nuWGD and 
    grows exponentially to a size of nuF
    
    Parameters:
        params (tuple): (T_WGD, nuWGD, nuF, gamma)

            - T_WGD: Time in the past at which the WGD occurred, creating the  
               allotetraploid population (in units of 2*Na generations).

            - nuWGD: Ratio of allotetraploid population immediately after WGD
                to ancient diploid population size (ratio of *census* sizes).

            - nuF: Ratio of contemporary allotetraploid population
                to ancient diploid population size (ratio of *census* sizes).

            - gamma: population-scaled selection coefficient (= 2*Na*s)
        ns (tuple): Sample sizes (n1,n2).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    T_WGD, nuWGD, nuF, gamma = params
    fs = bottlegrowth_sel((T_WGD, nuWGD, nuF, 0, gamma), ns, pts)
    return fs
bottlegrowth_noHE_sel.__param_names__ = ["T_WGD", "nuWGD", "nuF", "gamma"]

def three_epoch_sel(params, ns, pts):
    """
    Three epoch model of allotetraploid formation where the 
    allotetraploid population splits, maintains a size of nuWGD for T_WGD, 
    and then changes size again to nuF for a period of TF.
    This is similar to having a bottleneck for some period and then recover after the bottleneck.
    
    Parameters:
        params (tuple): (T_WGD, TF, nuWGD, nuF, H, gamma)

            - T_WGD: Time length between the WGD event and second size change, creating the  
               allotetraploid population (in units of 2*Na generations).

            - TF: Time in the past at which the second epoch begins.

            - nuWGD: Ratio of initial allotetraploid population (during first epoch)
                 to ancient diploid population size (ratio of *census* sizes).

            - nuF: Ratio of contemporary allotetraploid population (during second epoch)
                 to ancient diploid population size (ratio of *census* sizes).

            - H: homoeologous exchange rate (in terms of 2*Na*eta)

            - gamma: population-scaled selection coefficient (= 2*Na*s)
        ns (tuple): Sample sizes (n1,n2).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    T_WGD, TF, nuWGD, nuF, H, gamma  = params
    alloaflag = PolyInt.PloidyType.ALLOa
    allobflag = PolyInt.PloidyType.ALLOb
    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx, gamma=gamma)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    # T=1 divergence period between diploids
    phi = PolyInt.two_pops(phi, xx, 1, sel_dict1={"gamma": gamma}, sel_dict2={"gamma": gamma})
    # second epoch
    phi = PolyInt.two_pops(phi, xx, T_WGD, nu1=nuWGD, nu2=nuWGD, m12=H, m21=H,
                           sel_dict1={"gamma": gamma}, sel_dict2={"gamma": gamma},
                           ploidyflag1=alloaflag, ploidyflag2=allobflag)
    # third epoch
    phi = PolyInt.two_pops(phi, xx, TF, nu1=nuF, nu2=nuF, m12=H, m21=H,
                           sel_dict1={"gamma": gamma}, sel_dict2={"gamma": gamma},
                           ploidyflag1=alloaflag, ploidyflag2=allobflag)
    fs = Spectrum.from_phi(phi, ns, (xx,xx))
    return fs
three_epoch_sel.__param_names__ = ['T_WGD', 'TF', 'nuWGD', 'nuF', 'H', 'gamma']


def three_epoch_noHE_sel(params, ns, pts):
    """
    Three epoch model of allotetraploid formation where the 
    allotetraploid population splits, maintains a size of nuWGD for T_WGD, 
    and then changes size again to nuF for a period of TF.
    This is similar to having a bottleneck for some period and then recover after the bottleneck.
    
    Parameters:
        params (tuple): (T_WGD, TF, nuWGD, nuF, gamma)

            - T_WGD: Time length between the WGD event and second size change, creating the  
               allotetraploid population (in units of 2*Na generations).

            - TF: Time in the past at which the second epoch begins.

            - nuWGD: Ratio of initial allotetraploid population (during first epoch)
                 to ancient diploid population size (ratio of *census* sizes).

            - nuF: Ratio of contemporary allotetraploid population (during second epoch)
                 to ancient diploid population size (ratio of *census* sizes).

            - gamma: population-scaled selection coefficient (= 2*Na*s)
        ns (tuple): Sample sizes (n1,n2).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    T_WGD, TF, nuWGD, nuF, gamma = params
    fs = three_epoch_sel((T_WGD, TF, nuWGD, nuF, 0, gamma), ns, pts)
    return fs
three_epoch_noHE_sel.__param_names__ = ['T_WGD', 'TF', 'nuWGD', 'nuF', 'gamma']


# Note: I have not added the models with diploids because that requires 
# a 3D DFE and far too much computational power to be practical