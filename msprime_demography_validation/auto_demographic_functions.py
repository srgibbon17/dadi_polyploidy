from dadi import Numerics, PhiManip
from dadi.Spectrum_mod import Spectrum
from dadi.Polyploidy import Integration as PolyInt
import numpy

def auto_triplet_recurrent(params, ns, pts):
    """
    Model for a triplet of populations: two autotets and one diploid.
    Here, the autotetraploids form recurrently (i.e. split separately from the diploid).
    Each autotetraploid population follows a bottlegrowth model. 
    Also, the diploid population follows a 2 epoch model 
    with size changes fxed to the timing of the WGD events.
    
    Parameters:
        params (tuple): (T_WGD1, T_WGD2, nu_T1_WGD, nu_T1_c, nu_T2_WGD, nu_T2_c, M_tets, M_T_D)
        ns (tuple): Sample sizes (n1,n2,n3).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    T_WGD1, T_WGD2, nu_T1, nu_T2, M_tets, M_T_D = params
    
    # Here, T_WGD1 is the period between the first and second WGD events
    # and T_WGD2 is the period between the second WGD and the present

    autoflag = PolyInt.PloidyType.AUTO
    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    # first integration from first WGD event to the second WGD event
    phi = PolyInt.two_pops(phi, xx, T_WGD1, nu2=nu_T1, 
                           m21=M_T_D, ploidyflag2=autoflag)
    # second integration from second WGD event to present
    # note: f1 = 1 because the second tetraploid splits from the diploid
    phi = PhiManip.phi_2D_to_3D(phi, 1, xx, xx, xx)
    phi = PolyInt.three_pops(phi, xx, T_WGD1+T_WGD2, nu2=nu_T1, nu3=nu_T2,
                             m21=M_T_D, m31=M_T_D, m23=M_tets, m32=M_tets, 
                             ploidyflag2=autoflag, ploidyflag3=autoflag, initial_t=T_WGD1)
    fs = Spectrum.from_phi(phi, ns, (xx,xx,xx))
    return fs
auto_triplet_recurrent.__param_names__ = ['T_WGD1', 'T_WGD2', 'nu_T1', 'nu_T2', 'M_tets', 'M_T_D']

def auto_triplet_single_origin(params, ns, pts):
    """
    Model for a triplet of populations: two autotets and one diploid.
    Here, the autotetraploids form from a single origin 
        (i.e. the second tetraploid pop splits from the first).
    Each autotetraploid population follows a bottlegrowth model. 
    Also, the diploid population follows a 2 epoch model 
    with size changes fxed to the timing of the WGD events.

    
    Parameters:
        params (tuple): (T_WGD1, T_WGD2, nu_T1_WGD, nu_T1_c, nu_T2_WGD, nu_T2_c, M_tets, M_T_D)
        ns (tuple): Sample sizes (n1,n2,n3).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    T_WGD1, T_WGD2, nu_T1, nu_T2, M_tets, M_T_D = params
   
    # Here, T_WGD1 is the period between the first and second WGD events
    # and T_WGD2 is the period between the second WGD and the present 
   
    autoflag = PolyInt.PloidyType.AUTO
    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    # first integration from first WGD event to the second WGD event
    phi = PolyInt.two_pops(phi, xx, T_WGD1, nu2=nu_T1, 
                           m21=M_T_D, ploidyflag2=autoflag)
    # second integration from second WGD event to present
    # note: f1 = 0 because the second tetraploid splits from the first
    phi = PhiManip.phi_2D_to_3D(phi, 0, xx, xx, xx)
    phi = PolyInt.three_pops(phi, xx, T_WGD1+T_WGD2, nu2=nu_T1, nu3=nu_T2,
                             m21=M_T_D, m31=M_T_D, m23=M_tets, m32=M_tets, 
                             ploidyflag2=autoflag, ploidyflag3=autoflag, initial_t=T_WGD1)
    fs = Spectrum.from_phi(phi, ns, (xx,xx,xx))
    return fs
auto_triplet_single_origin.__param_names__ = ['T_WGD1', 'T_WGD2', 'nu_T1', 'nu_T2', 'M_tets', 'M_T_D']
