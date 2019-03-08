
# __all__ = ['EnergyCurrent']

import cython
#from operator import itemgetter
#import functools as ft
#import collections

###For Block _eval_hamil
#import numpy as np
import tinyarray as ta
#from scipy.sparse import coo_matrix

###FOR _is_herm
from libc cimport math

import kwant
import tkwant
cimport kwant.operator
# from kwant.operator cimport _LocalOperator
from kwant.graph.defs cimport gint
# from kwant cimport operator
from kwant.graph.defs import gint_dtype
#from kwant.system import InfiniteSystem
###FOR _check_ham
from kwant._common import UserCodeError, get_parameters


import numpy as np
# from libcpp.vector cimport vector
# gint_dtype = np.int32


def zerofct(*args, **kwargs):
    return 0

def add_two_lead_unit_cells(syst, lead, phase_func=zerofct):
    first_leaduc = tkwant.leads.add_voltage(syst, lead, phase_func)
    scnd_leaduc = tkwant.leads.add_voltage(syst, lead, zerofct)
    return first_leaduc, scnd_leaduc

class heatCurrentWithIc(kwant.operator._LocalOperator):
    r"""
    The heat current is given by I_h = I_E - mu * I_N + I_c,
    where I_E is the energy current, I_N is the particle current and
    I_c is an additional term (coupling) which is needed.

    An instance of this class can be called like a function to evaluate the
    expectation value with a wavefunction. See
    `~kwant.operator._LocalOperator.__call__` for details.

   Returns: :math:`0.5 * I_E - mu * I_N - 0.5 * \partial H/\partial t G_{0m} - 0.5 * I^E_{shifted}`
   for a given scattering state, i.e. for a given time, energy and alpha(=lead
   of scat state). Here, we use the explicit term of I_c which is
   :math:`I_c = -0.5 * I_E - 0.5 * \partial H/\partial t G_{0m} - 0.5 * I^E_{shifted}`.

   Parameters
   ----------
   syst : `~kwant.system.System`
   mu : chemical potential of the lead under investigation
   intracell_sites: A list of sites of the 1st unit cell of the lead.
       Either as `~kwant.builder.Site` or as Site of the finalized system
       (i.e. of type int).
   intercell_sites: A list of sites of the 2nd unit cell of the lead.
       Either as `~kwant.builder.Site` or as Site of the finalized system
       (i.e. of type int).
   tderiv_Hamil : the time derivative of the Hamiltonian. If unspecified, a
                  finite difference is used to approximate d/dt H
   del_t_deriv : small time-step used to calculate the difference quotient of H(t)

    """
    # cdef double mu
    # cdef EnergyCurrent energyCurrent
    # cdef kwant.operator.Current particleCurrent
    # cdef CurrentWithArbitHop tdepCoupling
    # cdef EnergyCurrent energyCurrentShifted

    @cython.embedsignature
    def __init__(self, syst, mu, intracell_sites, intercell_sites, *,
                 del_t_deriv=1e-3, tderiv_Hamil=None):
        r"""
        Initialize the different terms needed:
        EnergyCurrent, ParticleCurrent, explicite t-dep term and shifted I^E
        """
        self.mu = mu
        #Create instances of EnergyCurrent and Particle Current
        self.energyCurrent = LeadEnergyCurrent(syst, intracell_sites, intercell_sites, check_hermiticity=True)

        curr_where = self.energyCurrent.get_onwhere()
        self.particleCurrent = kwant.operator.Current(syst, onsite=1, where=curr_where, \
                 check_hermiticity=True, sum=True)

        # derivative of the Hamiltonian by finite difference
        def diff_Hamil(a,b,*args, **kwargs):
            time = args[0]
            retfunc = (syst.hamiltonian(a, b, time+del_t_deriv, *args[1:], **kwargs) \
                    - syst.hamiltonian(a, b, time, *args[1:], **kwargs))/del_t_deriv
            return retfunc
        # use d/dt H above if not otherwise specified
        if tderiv_Hamil == None:
            Hdot = diff_Hamil
        else:
            Hdot = tderiv_Hamil
        #Create instance of explicitely t-dep terms
        self.tdepCoupling = CurrentWithArbitHop(syst, onsite=1, \
                            arbit_hop_func=Hdot, where=curr_where, \
                            check_hermiticity=False, sum=True)

        #Create instance of of I^E_{shifted}
            # Create needed site-lists
        sitesAttachedToLead = _create_list_of_certain_neighbors(syst,
                                        intracell_sites, intercell_sites)
        neighSitesAttachedToLead = _create_list_of_certain_neighbors(syst,
                                        sitesAttachedToLead, intracell_sites)

        self.energyCurrentShifted = LeadEnergyCurrent(syst, sitesAttachedToLead,
                        neighSitesAttachedToLead, check_hermiticity=True)


    @cython.embedsignature
    def __call__(self, bra, ket=None, args=(), *, params=None):
        """
        Calculate the initialized operators given bra (and ket).
        """

        return_Ecurr = self.energyCurrent(bra, args = args, params=params)
        return_Ncurr = self.particleCurrent(bra, args = args, params=params)
        return_Ecurr_shift = self.energyCurrentShifted(bra, args = args, params=params)
        return_expl_time_dep = self.tdepCoupling(bra, args = args, params=params)
        if return_expl_time_dep.imag:
            assert(abs(return_expl_time_dep.imag) < 1e-14)
        return_expl_time_dep_real = return_expl_time_dep.real


        return - 0.5 * return_Ecurr + self.mu * return_Ncurr +\
               + 0.5 *  (return_Ecurr_shift - return_expl_time_dep_real)



class heatCurrentNoIc(kwant.operator._LocalOperator):
    """ The heatCurrent is given by I_h = I_E - mu * I_N + I_c, where I_E is the energy current, I_N is the particle current and I_c is an additional term which might be needed from a physical point of view, but is set to 0 in this class.

   Returns:  I_E - mu * I_N for a given scattering state, i.e. for a given time, energy and alpha(=lead and mode of scattering state).
    """

    @cython.embedsignature
    def __init__(self, syst, mu, intracell_sites, intercell_sites, *, check_hermiticity=True):
        """Initialize the different terms needed: EnergyCurrent and ParticleCurrent"""

        self.mu = mu
        #Create instances of EnergyCurrent and Particle Current
        self.energyCurrent = LeadEnergyCurrent(syst, intracell_sites, intercell_sites, check_hermiticity=True)

        curr_where = self.energyCurrent.get_onwhere()
        self.particleCurrent = kwant.operator.Current(syst, onsite=1, where=curr_where,
                 check_hermiticity=True, sum=True)


    @cython.embedsignature
    def __call__(self, bra, ket=None, args=(), *, params=None):
        """Calculate particle and energy current for a given bra (and ket). Like this, the scattering wave function in kwant has to be calculated only once per E and t."""

        return_Ecurr = self.energyCurrent(bra, args = args, params=params)
        return_Ncurr = self.particleCurrent(bra, args = args, params=params)

        return return_Ecurr - self.mu * return_Ncurr



class LeadEnergyCurrent(kwant.operator._LocalOperator):
    r"""An operator for calculating the energy current of into/from(?) a Lead.

    An instance of this class can be called like a function to evaluate the
    expectation value with a wavefunction. See
    `~kwant.operator.Current.__call__` for details.

    Parameters
    ----------
    syst : `~kwant.system.System`
    onsite : scalar or square matrix or dict or callable
        The onsite matrix that defines the density from which this current is
        derived. If a dict is given, it maps from site families to square
        matrices (scalars are allowed if the site family has 1 orbital per
        site). If a function is given it must take the same arguments as the
        onsite Hamiltonian functions of the system.
    intracell_sites: A list of sites of the 1st unit cell of the lead.
        Either as `~kwant.builder.Site` or as Site of the finalized system (i.e. of type int).
    intercell_sites: A list of sites of the 2nd unit cell of the lead.
        Either as `~kwant.builder.Site` or as Site of the finalized system (i.e. of type int).
    check_hermiticity : bool
        Check whether the provided ``onsite`` is Hermitian. If it
        is not Hermitian, then an error will be raised when the
        operator is evaluated.

    Notes
    -----
    We want to calculate the time-dependent energy current in analogy to the
    particle current which is given in tight-binding representation between
    site  :math:`i` and  :math:`j` by
      :math:`I_{ij}^N = -2\sum^\text{leads}_\alpha\int \frac{dE}{2\pi} f_\alpha(E)
      Im\left[ (\psi_i^{\alpha E})^\dagger(t)` H_{ij} \psi_j^{\alpha E}(t) \right]`,
    with the scattering states :math:`\psi^{\alpha E}_i(t)` from lead :math:`\alpha`
    at energy :math:`E` and time :math:`t`.
    For the current in the lead, one has to some over the cross section.

    The formula to calculate the energy current in a lead in tight-binding
    representation only changes the term in the energy integral of which the
    imaginary part is taken. It reads:
    Im\left[ \sum_{i,j \in \text{lead}} \sum_{q\in \bar{0}} (\psi_q^{\alpha E})^\dagger(t)` H_{qi} H_{ij} \psi_j^{\alpha E}(t) \right]`,
    where the sum over i,j runs over all sites in the corresponding lead and q
    runs over all sites in the scattering region :math:`\bar{0}`.
    Compared to the current, an additional Hamiltonian term is needed for the
    energy current.
    The sum over the lead is devided into two parts:
    The 'onSite' part, i.e. :math:`i==j` :
    Im\left[ \sum_{i \in \text{lead}} \sum_{q\in \bar{0}} (\psi_q^{\alpha E})^\dagger(t)` H_{qi} H_{ii} \psi_i^{\alpha E}(t) \right]`,
    which can be calculated with `~kwant.operator.Current` with an Onsite-term
    that is the Hamiltonian itself.
    The 'offSite' part, i.e. :math:`i\neq j`:
    Im\left[ \sum_{i\neq j \in \text{lead}} \sum_{q\in \bar{0}} (\psi_q^{\alpha E})^\dagger(t)` H_{qi} H_{ij} \psi_j^{\alpha E}(t) \right]`,
    which is calculated by the operator 'offEnergyCurrent' defined here.
    """
    @cython.embedsignature
    def __init__(self, syst, intracell_sites, intercell_sites, *, check_hermiticity=True):

        #check if site lists are a list of Sites or Integers(=finalized Sites) and make it list of Integers
        if isinstance(intracell_sites[0], kwant.builder.Site):
            intracell_sites_final = list(syst.id_by_site[s] for s in intracell_sites)
        else:
            assert(isinstance(intracell_sites[0],int))
            intracell_sites_final = intracell_sites
        if isinstance(intercell_sites[0], kwant.builder.Site):
            intercell_sites_final = list(syst.id_by_site[s] for s in intercell_sites)
        else:
            assert(isinstance(intercell_sites[0],int))
            intercell_sites_final = intercell_sites

        #where-lists creation
        onwhere, offwhere, _auxwhere_list, _wherepos_neigh, _auxpos_list = \
                    _create_where_lists_from_added_sites(syst,
                                        intracell_sites=intracell_sites_final,  intercell_sites=intercell_sites_final)

        auxwhere_list = np.asarray(_auxwhere_list, dtype=gint_dtype)
        wherepos_neigh = np.asarray(_wherepos_neigh, dtype=gint_dtype)
        auxpos_list = np.asarray(_auxpos_list, dtype=gint_dtype)

        del _auxwhere_list, _wherepos_neigh, _auxpos_list

        #create again `~kwant.builder.Site`-lists because of the method
        #'kwant.operator._normalize_hopping_where' which is called later
        self.onwhere = list((syst.sites[hop[0]],syst.sites[hop[1]]) for hop in onwhere)
        self.offwhere = list((syst.sites[hop[0]],syst.sites[hop[1]]) for hop in offwhere)

        #initialize 'offSite' term of Energy Current
        self.offSite = offEnergyCurrent(syst, self.offwhere, auxwhere_list,
                                        wherepos_neigh, auxpos_list,
                                        check_hermiticity=check_hermiticity, sum=True)

        #initialize 'onSite' term of Energy Current
        #and with the matching onSite-Hamiltonian function
        def onsiteHamil(a, *args, **params):
            if type(a) == kwant.builder.Site:
                a = syst.id_by_site[a]
            assert(type(a) == int)
            return syst.hamiltonian(a, a, *args, params=params)
        self.onSite = kwant.operator.Current(syst, onsite=onsiteHamil, where=self.onwhere, check_hermiticity=check_hermiticity, sum=True)

    def get_onwhere(self):
        return self.onwhere

    def get_offwhere(self):
        return self.offwhere


    @cython.embedsignature
    def __call__(self, bra, ket=None, args=(), *, params=None):
        r"""Calculate the energy current of a lead for both, :math:`i==j` and :math:`i\neq j` parts ( :math:`i` and  :math:`i` in lead)

        Parameters
        ----------
        bra, ket : sequence of complex
            Must have the same length as the number of orbitals
            in the system. If only one is provided, both ``bra``
            and ``ket`` are taken as equal.
        args : tuple, optional
            The arguments to pass to the system. Used to evaluate
            the ``onsite`` elements and, possibly, the system Hamiltonian.
            Mutually exclusive with 'params'.
        params : dict, optional
            Dictionary of parameter names and their values. Mutually exclusive
            with 'args'.

        Returns
        -------
        `float` that is the sum of 'onSite' and 'offSite' parts for a given bra.
        """

        resultoff = self.offSite(bra, ket, args=args, params=params)
        resulton = self.onSite(bra, ket, args=args, params=params)

        return resulton + resultoff


class LocalEnergyCurrent(kwant.operator._LocalOperator):
    r"""
    An operator for calculating the local energy current between two sites.

    An instance of this class can be called like a function to evaluate the
    expectation value with a wavefunction. See
    `~kwant.operator.Current.__call__` for details.

    Parameters
    ----------
    syst : `~kwant.system.System`
    onsite : scalar or square matrix or dict or callable
        The onsite matrix that defines the density from which this current is
        derived. If a dict is given, it maps from site families to square
        matrices (scalars are allowed if the site family has 1 orbital per
        site). If a function is given it must take the same arguments as the
        onsite Hamiltonian functions of the system.
    intracell_sites: A list of sites of the 1st unit cell of the lead.
        Either as `~kwant.builder.Site` or as Site of the finalized system (i.e. of type int).
    intercell_sites: A list of sites of the 2nd unit cell of the lead.
        Either as `~kwant.builder.Site` or as Site of the finalized system (i.e. of type int).
    check_hermiticity : bool
        Check whether the provided ``onsite`` is Hermitian. If it
        is not Hermitian, then an error will be raised when the
        operator is evaluated.

    Notes
    -----
    We want to calculate the time-dependent energy current in analogy to the
    particle current which is given in tight-binding representation between
    site  :math:`i` and  :math:`j` by
      :math:`I_{ij}^N = -2\sum^\text{leads}_\alpha\int \frac{dE}{2\pi} f_\alpha(E)
      Im\left[ (\psi_i^{\alpha E})^\dagger(t)` H_{ij} \psi_j^{\alpha E}(t) \right]`,
    with the scattering states :math:`\psi^{\alpha E}_i(t)` from lead :math:`\alpha`
    at energy :math:`E` and time :math:`t`.
    For the current in the lead, one has to some over the cross section.

    The formula to calculate the local energy currentin tight-binding
    representation only changes the term in the energy integral and reads:
    :math:`I_{ij}^E = -2\sum^\text{leads}_\alpha\int \frac{dE}{2\pi} f_\alpha(E)
     \sum_{k} 0.5* Im\left[ (\psi_k^{\alpha E})^\dagger(t)` H_{ki} H_{ij} \psi_j^{\alpha E}(t) - (\psi_k^{\alpha E})^\dagger(t)` H_{kj} H_{ji} \psi_i^{\alpha E}(t) \right]`,
    where i and j are the given sites where the local energy current is to be
    calculated and k runs formally over all sites in the whole system (effectively
    only k are all neighbors of i and j, respectively).
    Compared to the current, an additional Hamiltonian term is needed for the
    heat current.
    In total, we divide the imaginary part above into 4 terms. In the first term,
    k runs over all neighbors of i, whereas in the second term, k runs over all
    neighbors of j. Moreover, we distinguish for technical reasons the case
    `i==k` and `j==k`, which are calculated by the instances of onEnergyCurrent
    and all other cases by instances of offEnergyCurrent.
    """
    @cython.embedsignature
    def __init__(self, syst, where=None, *, check_hermiticity=True, sum=False):

        where_normalized = kwant.operator._normalize_hopping_where(syst, where)

        # extended where-lists creation and bookkeeping auxiliary lists
        offwhere_i, _auxwhere_list_i, _wherepos_neigh_i, _auxpos_list_i = \
                    _create_fullwhere_lists_for_local_ECurr(syst, where_normalized)

        auxwhere_list_i = np.asarray(_auxwhere_list_i, dtype=gint_dtype)
        wherepos_neigh_i = np.asarray(_wherepos_neigh_i, dtype=gint_dtype)
        auxpos_list_i = np.asarray(_auxpos_list_i, dtype=gint_dtype)

        del _auxwhere_list_i, _wherepos_neigh_i, _auxpos_list_i

        #initialize 'offSite' term of Energy Current
        offwhere_i_unfnlzd = list((syst.sites[hop[0]],syst.sites[hop[1]]) for hop in offwhere_i)
        self.offSite_i = offEnergyCurrent(syst, offwhere_i_unfnlzd, auxwhere_list_i,
                                        wherepos_neigh_i, auxpos_list_i,
                                        check_hermiticity=check_hermiticity, sum=sum)

        # initialize 'onSite' term of Energy Current
        # with the onSite-Hamiltonian
        def onsiteHamil(a, *args, **params):
            if type(a) == kwant.builder.Site:
                a = syst.id_by_site[a]
            assert(type(a) == int)
            return syst.hamiltonian(a, a, *args, params=params)
        self.onSite_i = kwant.operator.Current(syst, onsite=onsiteHamil, where=where, check_hermiticity=check_hermiticity, sum=sum)

        # the same for i and j in where swapped (to ensure I_E^ij == -I_E^ji,
        # in case of time-independent hopping ij)
        where_norm_swapped = [(hop[1],hop[0]) for hop in where_normalized]
        # extended where-lists creation and bookkeeping auxiliary lists
        offwhere_j, _auxwhere_list_j, _wherepos_neigh_j, _auxpos_list_j = \
                _create_fullwhere_lists_for_local_ECurr(syst, where_norm_swapped)
        auxwhere_list_j = np.asarray(_auxwhere_list_j, dtype=gint_dtype)
        wherepos_neigh_j = np.asarray(_wherepos_neigh_j, dtype=gint_dtype)
        auxpos_list_j = np.asarray(_auxpos_list_j, dtype=gint_dtype)

        del _auxwhere_list_j, _wherepos_neigh_j, _auxpos_list_j

        #initialize 'offSite' term of Energy Current
        offwhere_j_unfnlzd = list((syst.sites[hop[0]],syst.sites[hop[1]]) for hop in offwhere_j)
        self.offSite_j = offEnergyCurrent(syst, offwhere_j_unfnlzd, auxwhere_list_j,
                                        wherepos_neigh_j, auxpos_list_j,
                                        check_hermiticity=check_hermiticity, sum=sum)

        where_unfinalized = list((syst.sites[hop[0]],syst.sites[hop[1]]) for hop in where_norm_swapped)
        #initialize 'onSite' term of Energy Current
        self.onSite_j = kwant.operator.Current(syst, onsite=onsiteHamil, where=where_unfinalized, check_hermiticity=check_hermiticity, sum=sum)



    @cython.embedsignature
    def __call__(self, bra, ket=None, args=(), *, params=None):
        r"""Calculate the local energy current for both, :math:`i==j` and
        :math:`i\neq j` parts ( :math:`i` and  :math:`i` in lead)

        Parameters
        ----------
        bra, ket : sequence of complex
            Must have the same length as the number of orbitals
            in the system. If only one is provided, both ``bra``
            and ``ket`` are taken as equal.
        args : tuple, optional
            The arguments to pass to the system. Used to evaluate
            the ``onsite`` elements and, possibly, the system Hamiltonian.
            Mutually exclusive with 'params'.
        params : dict, optional
            Dictionary of parameter names and their values. Mutually exclusive
            with 'args'.

        Returns
        -------
        `float` that is the sum of 'onSite' and 'offSite' parts for a given bra.
        """

        resulton_i = self.onSite_i(bra, ket, args=args, params=params)
        resulton_j = self.onSite_j(bra, ket, args=args, params=params)
        assert len(resulton_i) == len(resulton_j)

        resultoff_i = self.offSite_i(bra, ket, args=args, params=params)
        resultoff_j = self.offSite_j(bra, ket, args=args, params=params)
        resultoff_i = np.resize(resultoff_i,len(resulton_i))
        resultoff_j = np.resize(resultoff_j,len(resulton_j))

        result = 0.5 * (resulton_i + resultoff_i - resulton_j - resultoff_j)

        return result



def flatten_2d_lists_with_bookkeeping(twodim_list):
    r"""
    Flattens the outer layer of a stacked list.

    Parameters
    ----------
        twodim_list: a stacked list

    Returns
    -------
        value_list: has the same elements as the initial 'twodim_list', but
                    the outermost layer is flattened
        auxlist: auxiliary list for bookeeping. Has the same shape as the
                 outermost 'twodim_list', where the elements indicate the starting
                 positions of the corresponding initial elements in the flattened
                 'value_list'.

    """
    value_list = []
    auxlist = [0] * (len(twodim_list)+1)

    count = 0
    for i, lst in enumerate(twodim_list):
        for value in lst:
            value_list.append(value)
            count += 1
        auxlist[i+1] = count

    return value_list, auxlist



def _create_fullwhere_lists_for_local_ECurr(fsyst, in_where):
    r"""
    Creates where lists from the hopping-list 'in_where' for local energy current calculation.

    Parameters
    ----------
    fsyst: finalized system under consideration

    in_where: list of all hoppings where the local energy current is to be
              calculated of finalized system (i.e. of type 'int')

    Returns
    -------
    lists with needed (additional) hoppings (tupels of 'int') and auxiliary lists for bookkeeping (where to find the needed hoppings):
    - offwhere: flattened list of all needed hoppings (not only ij, but also ik or jk)
    - auxwhere_list: list of the form '[0, start pos of additional hops,
                     # total hops]' for bookkeeping
    - wherepos_neigh_flat: list with information which initial hoppings ij are
                           connected with added hoppings jk (or ik)
    - auxpos_list: auxiliary list for bookkeeping of wherepos_neigh_flat
    """

    neigh_whereaux = []

    wherepos_neigh_stacked = []
    wherepos_neigh_dummy = []
    neigh_count = 0

    # list to be extended
    offwhere = [(hop[0],hop[1]) for hop in in_where]

    # find needed neighbors and store additional hoppings
    for i_idx, hop in enumerate(in_where):
        i = hop[0]
        j = hop[1]
        assert(type(i) == int or type(i) == np.int32)
        assert(type(j) == int or type(j) == np.int32)

        # neigh_whereaux.append([])
        wherepos_neigh_dummy = []

        # find neighbors of hop[0]
        for iedge in fsyst.graph.out_edge_ids(i):
            neighbor = fsyst.graph.head(iedge)
            # store new hopping
            neigh_whereaux.append((i,neighbor))
            # keep track of which added hoppings belong to initial hoppings
            wherepos_neigh_dummy.append(neigh_count)
            neigh_count += 1
        # for each hopping in where (ij), we have to store
        # the relative positions of the connected hoppings (ik)
        # in the extended flattened offwhere
        wherepos_neigh_stacked.append(wherepos_neigh_dummy)

    # append new hoppings to where
    offwhere = offwhere + neigh_whereaux
    # auxiliary list which tells us where added hoppings start in offwhere
    auxwhere_list = [0, len(in_where), len(offwhere)]

    # get flattened wherepos_neigh + auxiliary lists for bookkeeping
    wherepos_neigh_flat, auxpos_list = flatten_2d_lists_with_bookkeeping(wherepos_neigh_stacked)

    return offwhere, auxwhere_list, wherepos_neigh_flat, auxpos_list


def _create_where_lists_from_added_sites(fsyst, intracell_sites, intercell_sites):
    r"""Creates where lists from the sitelists 'intracell_sites' and 'intercell_sites' for lead energy current calculation.

    Parameters
    ----------
    intracell_sites: list of all sites of finalized system in 1st lead unit cell
                     (i.e. of type 'int', e.g. by fsyst.id_by_site[sites])

    intercell_sites: list of all sites of finalized system (i.e. of type 'int')
                     in 2nd lead unit cell

    Returns
    -------
    lists with needed hoppings (tupels of 'int') or indexes where to find the
    needed hoppings:
    - lead_scatreg_where: hoppings from scatregion to lead
    - where: all needed hoppings for lead energy current
    - auxwhere_list: list of the form '[0, start pos of additional hops,
                     # total hops]' for bookkeeping
    - wherepos_neigh_flat: list with information which lead-scat hoppings qi are
                           connected with lead-lead hoppings ij
    - auxpos_list: auxiliary list for bookkeeping of wherepos_neigh_flat
    OLD AND NOT NEEDED ANYMORE!!
    - where_idx: auxiliary stacked list to find the needed hopping in 'where'.
              It is defined such that where[where_idx[m_i][i][neigh]] gives back
              the hopping from site i in 1st lead unit cell to its neighbors
              'neigh' either in :
                    - m_i==0: the scattering region
                    - m_i==1: the 1st lead unit cell (i.e. from intracell_sites)
                    - m_i==2: the 2nd lead unit cell (i.e. from intercell_sites)
    """

    where = []
    lead_scatreg_where = []

    #auxlists to store hoppings
    central_whereaux = []
    lead_whereaux = []

    wherepos_neigh_stacked = []
    wherepos_neigh_dummy = []
    leadhop_count = 0

    # fill neighborlists; contains on purpose empty lists, if there is
    # no matching neighbor in the corresp. region
    for i_idx, i in enumerate(set(intracell_sites)):
        assert(type(i) == int)
        central_whereaux.append([])
        lead_whereaux.append([])
        wherepos_neigh_dummy = []
        for iedge in fsyst.graph.out_edge_ids(i):
            neighbor = fsyst.graph.head(iedge)
            #neighbor in lead
            if neighbor in set(intercell_sites+intracell_sites):
                lead_whereaux[i_idx].append((i,neighbor))
                wherepos_neigh_dummy.append(leadhop_count)
                leadhop_count += 1
            #neighbor in scattering region
            else:
                central_whereaux[i_idx].append((i,neighbor))

        # for each central-lead-hopping (iq), we have to store
        # the relative positions in the flattened where of the
        # lead-lead-hoppings (ij) with the same lead site 'i'
        for num_leadscat_hops in range(len(central_whereaux[i_idx])):
            wherepos_neigh_stacked.append(wherepos_neigh_dummy)


    # get flattened wheres, i.e. list of only tuples=hoppings, without the stacked structure
        # all hoppings
    where.append(central_whereaux)
    where.append(lead_whereaux)
    where = [tupel for qij in where for iSiteHoppings in qij for tupel in iSiteHoppings]
        # auxiliary list which tells us where added hoppings start in flattened where
    auxwhere_list = [0, len(central_whereaux), len(where)]
        # hoppings from scat-region to lead
    lead_scatreg_where = [tupel for iSiteHoppings in central_whereaux for tupel in iSiteHoppings]

        # get flattened wherepos_neigh + auxiliary lists for bookkeeping
    wherepos_neigh_flat, auxpos_list = flatten_2d_lists_with_bookkeeping(wherepos_neigh_stacked)

    return lead_scatreg_where, where, auxwhere_list, wherepos_neigh_flat, auxpos_list



def _create_list_of_certain_neighbors(fsyst, initial_list, forbidden_list):
    r"""
    Creates a list of sites, which are neighbors the sites in 'initial_list'
    but which are neither in 'forbidden_list' nor in 'initial_list'.
    Used for the shifted energy current in the heat current.

    Parameters
    ----------
    initial_list: list of sites, either as `int` or `~kwant.builder.Site`
    finitial_list: list of sites, either as `int` or `~kwant.builder.Site`

    Returns
    -------
    list of sites as `int`
    """
    #check type of sites in the given lists and convert to int if needed
    if isinstance(initial_list[0], kwant.builder.Site):
        initial_list = list(fsyst.id_by_site[s] for s in initial_list)
    if isinstance(forbidden_list[0], kwant.builder.Site):
        forbidden_list = list(fsyst.id_by_site[s] for s in forbidden_list)
    assert type(initial_list[0]) == int
    assert type(forbidden_list[0]) == int

    # create list in which the neighbors of 'initial_list' which are not in
    # 'forbidden_list' nor in 'initial_list' are stored.
    neighbor_list = []
    for i in initial_list:
        assert(type(i) == int)
        for iedge in fsyst.graph.out_edge_ids(i):
            neighbor = fsyst.graph.head(iedge)
            #neighbor in forbidden_list -> do nothing:
            if neighbor in set(forbidden_list):
                pass
            #neighbor in initial_list -> do nothing:
            elif neighbor in set(initial_list):
                pass
            #neighbor already in neighbor_list -> do nothing:
            elif neighbor in set(neighbor_list):
                pass
            #neighbor not yey in neighbor_list -> add it
            else:
                neighbor_list.append(neighbor)

    return neighbor_list





# supported operations within the `_operate` method
ctypedef enum operation:
    MAT_ELS
    ACT



cdef class offEnergyCurrent(kwant.operator._LocalOperator):
    r"""An operator for calculating the part of the energy currents with two
    hopping Hamiltonians.

    An instance of this class can be called like a function to evaluate the
    expectation value with a wavefunction. See
    `~kwant.operator._LocalOperator.__call__` and `~kwant.operator.offEnergyCurrent._operate` for details.

    Parameters
    ----------
    syst : `~kwant.system.System`
    where : sequence of pairs of `int`
    auxwhere_list: auxiliary list for bookkeeping which hoppings in where belong
                   to the first and which to the second Hamiltonian
    wherepos_neigh: list with information which hoppings of first Hamiltonian are
                         connected to which hoppings of the second Hamiltonian
    auxpos_list: auxiliary list for bookkeeping of wherepos_neigh_flat

    OUTDATED!where_idx: stacked list which is used to find the right hopping in 'where'

    check_hermiticity : bool
        Check whether the provided ``onsite`` is Hermitian. If it
        is not Hermitian, then an error will be raised when the
        operator is evaluated. (NOTE: Probably not necessary here.)
    sum : bool, default: False
        If True, then calling this operator will return a single scalar,
        otherwise a vector will be returned. For the Lead energy current, it is
        set to true.

    Notes
    -----
    To be evaluated: :math:`\sum_{ijq} [ -2 Im\{ bra^\ast_{q} H_{qi} H_{ij} ket_{j}  \}]`,
    where :math:`q, i and j` are sites.

    TODO: The symbols q, i and j for sites are in agreement with our definition
          of the lead energy current, but not with the local energy current.
          Change definitions of either lead heat current or local energy current.
    """

    cdef public gint[:] auxwhere_list, wherepos_neigh, auxpos_list

    @cython.embedsignature
    def __init__(self, syst, where, auxwhere_list,
                                    wherepos_neigh, auxpos_list, *, check_hermiticity=True, sum=True):
        assert(where != None)
        self.auxwhere_list = auxwhere_list
        self.wherepos_neigh = wherepos_neigh
        self.auxpos_list = auxpos_list

        where = kwant.operator._normalize_hopping_where(syst, where)
        super().__init__(syst, onsite=1, where=where, check_hermiticity=check_hermiticity, sum=sum)

    @cython.embedsignature
    def bind(self, args=(), *, params=None):
        """Bind the given arguments to this operator.

        Returns a copy of this operator that does not need to be passed extra
        arguments when subsequently called or when using the ``act`` method.
        """
        q = super().bind(args, params=params)
        q.auxwhere_list = self.auxwhere_list
        q.wherepos_neigh = self.wherepos_neigh
        q.auxpos_list = self.auxpos_list
        q._bound_hamiltonian = self._eval_hamiltonian(args, params)
        return q

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _operate(self, complex[:] out_data, complex[:] bra, complex[:] ket,
                 args, operation op, *, params=None):
        # prepare onsite matrices and hamiltonians
        cdef int unique_onsite = not callable(self.onsite)
        cdef complex *H_ab = NULL
        cdef kwant.operator.BlockSparseMatrix H_ab_blocks

        assert(unique_onsite)

        if self._bound_hamiltonian:
            H_ab_blocks = self._bound_hamiltonian
        else:
            H_ab_blocks = self._eval_hamiltonian(args, params)

        # main loop
        cdef gint q_s, q_norbs, j_s, j_norbs, i_norbs
        cdef gint iq_pos, ij_pos, num_ijneighs, num_iqhops, neigh
        cdef gint o_q, o_i, o_j
        cdef complex tmp
        if op == ACT:
            raise NotImplementedError()
        elif op == MAT_ELS:
            num_iqhops = self.auxwhere_list[1]
            for iq_pos in range(num_iqhops):
                # get Hamiltonian H_qi as well as
                # WF start index q_s and number of orbitals
                H_iq = H_ab_blocks.get(iq_pos)
                q_s = H_ab_blocks.block_offsets[iq_pos, 1]
                q_norbs = H_ab_blocks.block_shapes[iq_pos, 1]
                i_norbs = H_ab_blocks.block_shapes[iq_pos, 0]
                num_ijneighs = self.auxpos_list[iq_pos+1] - self.auxpos_list[iq_pos]
                # initialize sum variable
                tmp = 0
                for neigh in range(num_ijneighs):
                    # get Hamiltonian H_ij as well as
                    # WF start index j_s and number of orbitals
                    ij_pos = num_iqhops + self.wherepos_neigh[self.auxpos_list[iq_pos]+neigh]
                    H_ij = H_ab_blocks.get(ij_pos)
                    j_s = H_ab_blocks.block_offsets[ij_pos, 1]
                    j_norbs = H_ab_blocks.block_shapes[ij_pos, 1]
                    assert(i_norbs == H_ab_blocks.block_shapes[ij_pos, 0])
                    ### do the actual calculation
                    for o_q in range(q_norbs):
                        for o_i in range(i_norbs):
                            for o_j in range(j_norbs):
                                tmp += (bra[q_s+o_q].conjugate() *
                                       H_iq[o_i*q_norbs + o_q].conjugate() *
                                       H_ij[o_i*j_norbs + o_j] *
                                       ket[j_s+o_j]
                                     - bra[j_s+o_j].conjugate() *
                                       H_ij[o_i*j_norbs + o_j].conjugate() *
                                       H_iq[o_i*q_norbs + o_q] *
                                       ket[q_s+o_q])
                # save result
                out_data[iq_pos] = 1j * tmp



cdef class CurrentWithArbitHop(kwant.operator._LocalOperator):
    r"""An operator for calculating general currents with arbitrary hopping.

    An instance of this class can be called like a function to evaluate the
    expectation value with a wavefunction. See
    `~kwant.operator.Current.__call__` for details.

    Parameters
    ----------
    syst : `~kwant.system.System`
    onsite : scalar or square matrix or dict or callable
        The onsite matrix that defines the density from which this current is
        derived. If a dict is given, it maps from site families to square
        matrices (scalars are allowed if the site family has 1 orbital per
        site). If a function is given it must take the same arguments as the
        onsite Hamiltonian functions of the system.
    arbit_hop_func: function which takes the same parameters as the Hamiltonians
                    (2 sites + args or params), which replaces the Hamiltonian
                    in the 'standard current'
    where : sequence of pairs of `int` or `~kwant.builder.Site`, or callable, optional
        Where to evaluate the operator. If ``syst`` is not a finalized Builder,
        then this should be a sequence of pairs of integers. If a function is
        provided, it should take a pair of integers or a pair of
        `~kwant.builder.Site` (if ``syst`` is a finalized builder) and return
        True or False.  If not provided, the operator will be calculated over
        all hoppings in the system.
    check_hermiticity : bool
        Check whether the provided ``onsite`` is Hermitian. If it
        is not Hermitian, then an error will be raised when the
        operator is evaluated.
    sum : bool, default: False
        If True, then calling this operator will return a single scalar,
        otherwise a vector will be returned (see
        `~kwant.operator.Current.__call__` for details).

    Notes
    -----
    In general, this class is used to calculate the matrix products:
    :math:`2\sum^\text{leads}_\alpha\int \frac{dE}{2\pi} f_\alpha(E)
      Re\left[ (\psi_i^{\alpha E})^\dagger(t)` M_i O_{ij} \psi_j^{\alpha E}(t) \right]`,
    where  :math:`M_i` is an arbitrary onsite term and O_{ij} is an
    arbitrary hopping term. Note that a major difference is that the REAL part
    of the wavefunctions and potentials is used instead of the IMAGINARY part
    as it is done in the current.

    This class is mostly needed for explicite time-dependendence of hoppings
    for heat current in a lead:
    :math:`\psi_j^\dagger  \partial H_{ij} /\partial t \psi_i
    """

    cdef public object arbit_hop_func, _bound_arbit_hop_func

    @cython.embedsignature
    def __init__(self, syst, onsite=1, arbit_hop_func=0, where=None,
                 *, check_hermiticity=False, sum=True):
        where = kwant.operator._normalize_hopping_where(syst, where)
        assert callable(arbit_hop_func)
        self.arbit_hop_func = arbit_hop_func
        super().__init__(syst, onsite, where,
                         check_hermiticity=check_hermiticity, sum=sum)


    cdef kwant.operator.BlockSparseMatrix _eval_arbit_hop_func(self, args, params):
        """Evaluate the onsite matrices on all elements of `where`"""
        assert callable(self.arbit_hop_func)
        assert not (args and params)
        params = params or {}
        matrix = ta.matrix
        arbit_hop_func = self.arbit_hop_func
        check_hermiticity = self.check_hermiticity
        hamiltonian = self.syst.hamiltonian

         #XXX: Checks for sanity of 'arbit_hop_func' are missing
         # required, defaults, takes_kw = self._onsite_params_info
         # invalid_params = set(params).intersection(set(defaults))
         # if invalid_params:
         #     raise ValueError("Parameters {} have default values "
         #                      "and may not be set with 'params'"
         #                      .format(', '.join(invalid_params)))
         #
         # if params and not takes_kw:
         #     params = {pn: params[pn] for pn in required}

        def get_arbit_hop_func(a, a_norbs, b, b_norbs):
            mat = matrix(arbit_hop_func(a, b, *args, params=params), complex)
            # kwant.operator.
            _check_ham(mat, hamiltonian, args, params,
                       a, a_norbs, b, b_norbs, check_hermiticity)
            return mat

        offsets, norbs = kwant.operator._get_all_orbs(self.where, self._site_ranges)
        return kwant.operator.BlockSparseMatrix(self.where, offsets, norbs, get_arbit_hop_func)


    @cython.embedsignature
    def bind(self, args=(), *, params=None):
        """Bind the given arguments to this operator.

        Returns a copy of this operator that does not need to be passed extra
        arguments when subsequently called or when using the ``act`` method.
        """
        q = super().bind(args, params=params)
        q._bound_arbit_hop_func = self._eval_arbit_hop_func(args, params)
        return q

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _operate(self, complex[:] out_data, complex[:] bra, complex[:] ket,
                 args, operation op, *, params=None):
        # prepare onsite matrices and arbitrary hopping functions
        cdef int unique_onsite = not callable(self.onsite)
        cdef complex[:, :] _tmp_mat
        cdef complex *M_a = NULL
        cdef complex *hopfunc_ab = NULL
        cdef kwant.operator.BlockSparseMatrix M_a_blocks, hopfunc_ab_blocks

        if unique_onsite:
            _tmp_mat = self.onsite
            M_a = <complex*> &_tmp_mat[0, 0]
        elif self._bound_onsite:
            M_a_blocks = self._bound_onsite
        else:
            M_a_blocks = self._eval_onsites(args, params)

        if self._bound_arbit_hop_func:
            hopfunc_ab_blocks = self._bound_arbit_hop_func
        else:
            hopfunc_ab_blocks = self._eval_arbit_hop_func(args, params)

        # main loop
        cdef gint a, a_s, a_norbs, b, b_s, b_norbs
        cdef gint i, j, k, w
        cdef complex tmp
        for w in range(self.where.shape[0]):
            ### get the next hopping's start orbitals and numbers of orbitals
            a_s = hopfunc_ab_blocks.block_offsets[w, 0]
            b_s = hopfunc_ab_blocks.block_offsets[w, 1]
            a_norbs = hopfunc_ab_blocks.block_shapes[w, 0]
            b_norbs = hopfunc_ab_blocks.block_shapes[w, 1]
            ### get the next onsite and hopfunc matrices
            hopfunc_ab = hopfunc_ab_blocks.get(w)
            if not unique_onsite:
                M_a = M_a_blocks.get(w)
            ### do the actual calculation
            if op == MAT_ELS:
                tmp = 0
                for i in range(b_norbs):
                    for j in range(a_norbs):
                        for k in range(a_norbs):
                            tmp += (bra[b_s + i].conjugate() *
                                    hopfunc_ab[j * b_norbs + i].conjugate() *
                                    M_a[j * a_norbs + k] * ket[a_s + k]
                                   + bra[a_s + j].conjugate() *
                                    M_a[j * a_norbs + k] *
                                    hopfunc_ab[k * b_norbs + i] * ket[b_s + i])
                out_data[w] = tmp
            elif op == ACT:
                raise NotImplementedError()
                # for i in range(b_norbs):
                #     for j in range(a_norbs):
                #         for k in range(a_norbs):
                #             out_data[b_s + i] = (
                #                 out_data[b_s + i] +
                #                 1j * hopfunc_ab[j * b_norbs + i].conjugate() *
                #                 M_a[j * a_norbs + k] * ket[a_s + k])
                #             out_data[a_s + j] = (
                #                 out_data[a_s + j] -
                #                 1j * M_a[j * a_norbs + k] * hopfunc_ab[k * b_norbs + i] *
                #                 ket[b_s + i])





#
#
# cdef class BlockSparseMatrix2:
#     """A sparse matrix stored as dense blocks.
#
#     Parameters
#     ----------
#     where : gint[:, :]
#         ``Nx2`` matrix or ``Nx1`` matrix: the arguments ``a``
#         and ``b`` to be used when evaluating ``f``. If an
#         ``Nx1`` matrix, then ``b=a``.
#     block_offsets : gint[:, :]
#         The row and column offsets for the start of each block
#         in the sparse matrix: ``(row_offset, col_offset)``.
#     block_shapes : gint[:, :]
#         ``Nx2`` array: the shapes of each block, ``(n_rows, n_cols)``.
#     f : callable
#         evaluates matrix blocks. Has signature ``(a, n_rows, b, n_cols)``
#         where all the arguments are integers and
#         ``a`` and ``b`` are the contents of ``where``. This function
#         must return a matrix of shape ``(n_rows, n_cols)``.
#
#     Attributes
#     ----------
#     block_offsets : gint[:, :]
#         The row and column offsets for the start of each block
#         in the sparse matrix: ``(row_offset, col_offset)``.
#     block_shapes : gint[:, :]
#         The shape of each block: ``(n_rows, n_cols)``
#     data_offsets : gint[:]
#         The offsets of the start of each matrix block in `data`.
#     data : complex[:]
#         The matrix of each block, stored in row-major (C) order.
#     """
#
#     cdef public int[:, :] block_offsets, block_shapes
#     cdef public int[:] data_offsets
#     cdef public complex[:] data
#
#     @cython.embedsignature
#     @cython.boundscheck(False)
#     @cython.wraparound(False)
#     def __init__(self, int[:, :] where, int[:, :] block_offsets,
#                   int[:, :] block_shapes, f):
#         if (block_offsets.shape[0] != where.shape[0] or
#             block_shapes.shape[0] != where.shape[0]):
#             raise ValueError('Arrays should be the same length along '
#                              'the first axis.')
#         self.block_shapes = block_shapes
#         self.block_offsets = block_offsets
#         self.data_offsets = np.empty(where.shape[0], dtype=gint_dtype)
#         ### calculate shapes and data_offsets
#         cdef int w, data_size = 0
#         for w in range(where.shape[0]):
#             self.data_offsets[w] = data_size
#             data_size += block_shapes[w, 0] * block_shapes[w, 1]
#         ### Populate data array
#         self.data = np.empty((data_size,), dtype=complex)
#         cdef complex[:, :] mat
#         cdef int i, j, off, a, b, a_norbs, b_norbs
#         for w in range(where.shape[0]):
#             off = self.data_offsets[w]
#             a_norbs = self.block_shapes[w, 0]
#             b_norbs = self.block_shapes[w, 1]
#             a = where[w, 0]
#             b = a if where.shape[1] == 1 else where[w, 1]
#             # call the function that gives the matrix
#             mat = f(a, a_norbs, b, b_norbs)
#             # Copy data
#             for i in range(a_norbs):
#                 for j in range(b_norbs):
#                     self.data[off + i * b_norbs + j] = mat[i, j]
#
#     cdef complex* get(self, int block_idx):
#         return  <complex*> &self.data[0] + self.data_offsets[block_idx]
#
#     def __getstate__(self):
#         return tuple(map(np.asarray, (
#             self.block_offsets,
#             self.block_shapes,
#             self.data_offsets,
#             self.data
#         )))
#
#     def __setstate__(self, state):
#         (self.block_offsets,
#          self.block_shapes,
#          self.data_offsets,
#          self.data,
#         ) = state
#





_herm_msg = ('{0} matrix is not hermitian, use the option '
            '`check_hermiticity=True` if this is intentional.')
_shape_msg = ('{0} matrix dimensions do not match '
     'the declared number of orbitals')


# cdef int _check_onsite2(complex[:, :] M, int norbs,
#                        int check_hermiticity) except -1:
#     "Check onsite matrix for correct shape and hermiticity."
#     if M.shape[0] != M.shape[1]:
#         raise UserCodeError('Onsite matrix is not square')
#     if M.shape[0] != norbs:
#         raise UserCodeError(_shape_msg.format('Onsite'))
#     if check_hermiticity and not _is_herm_conj(M, M):
#         raise ValueError(_herm_msg.format('Onsite'))
#     return 0


cdef int _check_ham(complex[:, :] H, ham, args, params,
                    int a, int a_norbs, int b, int b_norbs,
                    int check_hermiticity) except -1:
    "Check Hamiltonian matrix for correct shape and hermiticity."
    if H.shape[0] != a_norbs and H.shape[1] != b_norbs:
        raise UserCodeError(kwant.operator._shape_msg.format('Hamiltonian'))
    if check_hermiticity:
        # call the "partner" element if we are not on the diagonal
        H_conj = H if a == b else ta.matrix(ham(b, a, *args, params=params),
                                                complex)
        if not _is_herm_conj(H_conj, H):
            raise ValueError(kwant.operator._herm_msg.format('Hamiltonian'))
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _is_herm_conj(complex[:, :] a, complex[:, :] b,
                       double atol=1e-300, double rtol=1e-13) except -1:
    "Return True if `a` is the Hermitian conjugate of `b`."
    assert a.shape[0] == b.shape[1]
    assert a.shape[1] == b.shape[0]

    # compute max(a)
    cdef double tmp, max_a = 0
    cdef int i, j
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            tmp = a[i, j].real * a[i, j].real + a[i, j].imag * a[i, j].imag
            if tmp > max_a:
                max_a = tmp
    max_a = math.sqrt(max_a)

    cdef double tol = rtol * max_a + atol
    cdef complex ctmp
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            ctmp = a[i, j] - b[j, i].conjugate()
            tmp = ctmp.real * ctmp.real + ctmp.imag * ctmp.imag
            if tmp > tol:
                return False
    return True
