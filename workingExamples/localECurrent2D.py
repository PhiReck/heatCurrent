# Quantum dot with a time-dependent lead voltage

import numpy as np
from math import sqrt
import cmath
import matplotlib.pyplot as plt
import kwant
import tkwant
import operatorsHeatCurrent
from mpi4py import MPI
import time as timer
import os
import tinyarray
import testLocalECurrent.operatorsHeatCurrentWherePosNeighLocal


tau_x = tinyarray.array([[0, 1], [1, 0]])
tau_y = tinyarray.array([[0, -1j], [1j, 0]])
tau_z = tinyarray.array([[1, 0], [0, -1]])


# True if it is the master process
def am_master():
    return MPI.COMM_WORLD.rank == 0


# Create necessary directories if not present
if am_master():
    directories = ['data', 'Ih_plots']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)


# Create the system #######################################################
###########################################################################
#def make_system(a, t, params):
#    # Create the Bravais lattice and the system
#    lat = kwant.lattice.chain(a, norbs=1)
#    syst = kwant.Builder()

#    # Define the region of study: add the time-dependent potential
#    @tkwant.time_dependent
#    def qdot_hopping(site1, site2, time, params=params):
#        return -params['tc'] * faraday_flux_direct(time, params)

#    # Create onsite terms
#    syst[(lat(0))] = params['energyOffset']
#    syst[(lat(1))] = params['energyOffset'] + params['eps_c']
#    syst[(lat(2))] = params['energyOffset']
#    syst[(lat(-1))] = params['energyOffset']
#    syst[(lat(3))] = params['energyOffset']

#    # Create hopping terms
#    syst[(lat(0), lat(1))] = qdot_hopping
#    syst[(lat(2), lat(1))] = -params['tc']
#    syst[(lat(-1), lat(0))] = -t
#    syst[(lat(2), lat(3))] = -t

#    # Define and attach the leads
#    lead = kwant.Builder(kwant.TranslationalSymmetry((-a,)))
#    lead[lat(0)] = params['energyOffset']
#    lead[lat.neighbors()] = -t
#    syst.attach_lead(lead)
#    syst.attach_lead(lead.reversed())

#    return syst, lat, lead


def make_system(a=1, W=2, L=6, barrier=1.5, barrierpos=(3, 4),
                mu=0.4, Delta=0.1, Deltapos=4, t=1.0, phs=True):
    # Start with an empty tight-binding system. On each site, there
    # are now electron and hole orbitals, so we must specify the
    # number of orbitals per site. The orbital structure is the same
    # as in the Hamiltonian.
    lat = kwant.lattice.square(norbs=2)
    syst = kwant.Builder()

    #### Define the scattering region. ####
    # The superconducting order parameter couples electron and hole orbitals
    # on each site, and hence enters as an onsite potential.
    # The pairing is only included beyond the point 'Deltapos' in the scattering region.
    for y in range(int(W)):
        for x in range(Deltapos):
            syst[lat(x, y)] = (4 * t - mu) * tau_z + 0.1*x*tau_y + 0.2 * y * tau_x
        for x in range(Deltapos, L):
            syst[lat(x, y)] = (4 * t - mu) * tau_z + Delta * tau_x + 0.1*x*tau_y + 0.2 * y * tau_x

    # The tunnel barrier
    syst[(lat(x, y) for x in range(barrierpos[0], barrierpos[1])
         for y in range(W))] = (4 * t + barrier - mu) * tau_z

    # Hoppings
    syst[lat.neighbors()] = -t * tau_z #+ 0.1*x*tau_y + 0.2 * y * tau_x
    #### Define the leads. ####
    # Left lead - normal, so the order parameter is zero.
    sym_left = kwant.TranslationalSymmetry((-a, 0))
    # Specify the conservation law used to treat electrons and holes separately.
    # We only do this in the left lead, where the pairing is zero.
    lead0 = kwant.Builder(sym_left, conservation_law=-tau_z, particle_hole=tau_y)
    lead0[(lat(0, j) for j in range(W))] = (4 * t - mu) * tau_z
    lead0[lat.neighbors()] = -t * tau_z
    # Right lead - superconducting, so the order parameter is included.
    sym_right = kwant.TranslationalSymmetry((a, 0))
    lead1 = kwant.Builder(sym_right)
    lead1[(lat(0, j) for j in range(W))] = (4 * t - mu) * tau_z + Delta * tau_x
    lead1[lat.neighbors()] = -t * tau_z

    #### Attach the leads and return the system. ####
    syst.attach_lead(lead0)
    syst.attach_lead(lead1)
    
#    lead = lead1.finalized()
#    kwant.plotter.bands(lead, show=False)
    
    return syst, lat


# Varying potential of the lead ####################################
###########################################################################
# for direct use in make_system
def faraday_flux_direct(time, *args, **kwargs):
    t_start = args[0]['t0']
    if time <= t_start:
        return 1
    else:
        return cmath.exp(1j*args[0]['V_lead'] * (time - t_start))


# for use in tkwant.add_voltage
def faraday_flux(time, *args, **kwargs):
    t_start = args[0]['t0']
    if time <= t_start:
        return 0
    else:
        return args[0]['V_lead'] * (time - t_start)


# Useful functions ########################################################
###########################################################################

def write_file(array, file_name):
    # To write data (a 2D array) into a file
    data = open(file_name, 'w')
    for j in array:
        for i in j:
            data.write(str(i) + ' ')
        data.write('\n')
    data.close()


def plot_graph(results):
    # To plot the figure as a graph
    fig, ax = plt.subplots(nrows=1, ncols=1)  # Create the subplot

    # Probably, it would be easier to use transpose of results instead
    curve = np.array([])
    times = np.array([])
    for i in range(len(results)):
        times = np.append(times, results[i][0])
        curve = np.append(curve, results[i][1])

    ax.plot(times, curve)

    ax.set_title('Quantum dot : heatcurrent vs time')
    ax.set_ylabel('I_h tau/V')
    ax.set_xlabel('t  / tau')

    return fig



def plot_currents(syst, currents):
    fig, axes = plt.subplots(1, len(currents))
    if not hasattr(axes, '__len__'):
        axes = (axes,)
    for ax, (title, current) in zip(axes, currents):
        kwant.plotter.current(syst, current, ax=ax, colorbar=False)
        ax.set_title(title)
    plt.show()
    
    
def savefig_currents(syst, currents, count):
    fig, axes = plt.subplots(1, len(currents))
    if not hasattr(axes, '__len__'):
        axes = (axes,)
    for ax, (title, current) in zip(axes, currents):
        kwant.plotter.current(syst, current, ax=ax, colorbar=False)
        ax.set_title(title)
    dirpath = os.getcwd()
    filename = dirpath + '/pic/2d' + str(count) + '.png'

    fig.savefig(filename)
#    plt.show()



# Calculation over the system #############################################
###########################################################################
def manybody(syst, times, boundaries, occupations, integration_regions,
             integration_error, operator, params):
    # Create the solver
    S = tkwant.manybody.Solver(syst, boundaries, occupations,
                               integration_regions,
                               integration_error=integration_error,
                               args=(params, ))
    count = 0
    # Evolve the system forward in time, and calculate the heat current
    results = []
    start = timer.clock()
    for time in times:
        tmp_result = S(operator, time)
        results.append(tmp_result)
        # Console output and save local current
        if am_master():
            plot_currents(syst, [('$J_{E}$', tmp_result)]) 
#            savefig_currents(syst, [('$J_{E}$', tmp_result)], count) 
#            count += 1
            print(str(int(timer.clock() - start)) + ' ' + 't = ' + str(time)
                  + ', current = ' + str(results[-1]))

    return results


def calculations(syst, lat, ilead, tmax, t, boundaries_type, num_cells,
                 strength, degree, times, integration_error, params):


    # Finalize the system
    tkwant.leads.add_voltage(syst, 0, faraday_flux)
    syst = syst.finalized()
    operator = operatorsHeatCurrent.LocalEnergyCurrent(syst, where=None)

    # Boundary conditions
    if boundaries_type == 'simple_boundaries':
        boundaries = [tkwant.leads.SimpleBoundary(max_time=tmax)
                      for lead in syst.leads]
    elif boundaries_type == 'absorbing_boundaries':
        boundaries = \
         [tkwant.leads.MonomialAbsorbingBoundary(num_cells=num_cells,
                                                 strength=strength,
                                                 degree=degree) for l in syst.leads]
    else:
        raise ValueError("Ill-defined boundary conditions in the leads")

    # Energy integration region
    band_upperE = 6.5 #params['energyOffset'] + 2 * t
    band_lowerE = 0.61 #params['energyOffset'] - 2 * t
    integration_regions = [(0, 0, (1e-6 + band_lowerE, band_upperE - 1e-6)),
                           (1, 0, (1e-6 + band_lowerE, band_upperE - 1e-6))]

    # Occupation, for each lead
    occupations = [tkwant.manybody.fermi_dirac(params['muL'], params['TL']),
                   tkwant.manybody.fermi_dirac(params['muR'], params['TR'])]

    # Calculate the heat Current
    results = manybody(syst, times, boundaries, occupations,
                       integration_regions, integration_error, operator,
                       params)

    return results


# Main ####################################################################
###########################################################################
def main(a, t, ilead, boundaries_type, num_cells, strength, degree, tmax, dt,
         integration_error, plotSystem, showFigure, showPlots, params):
    # File name basis: used to write the data files
    file_name_basis = 'lead=' + str("{:.3f}".format(ilead)) + \
                      '_tc=' + str("{:.3f}".format(params['tc'])) + \
                      '_t0=' + str("{:.3f}".format(params['t0'])) + \
                      '_tmax=' + str("{:.3f}".format(tmax)) + \
                      '_dt=' + str("{:.3f}".format(dt)) + \
                      '_muL=' + str("{:.3f}".format(params['muL'])) + \
                      '_muR=' + str("{:.3f}".format(params['muR'])) + \
                      '_TL=' + str("{:.3f}".format(params['TL'])) + \
                      '_TR=' + str("{:.3f}".format(params['TR'])) + \
                      '_epsc=' + str("{:.3f}".format(params['eps_c'])) + \
                      '_V_lead=' + str("{:.3f}".format(params['V_lead'])) + \
                      '_lam=' + str("{:.3f}".format(params['lam']))

    times = np.arange(0, tmax, dt)

    # Simulation ######################################################
    ###################################################################

    # Create the initial quantum dot system
#    syst, lat, lead = make_system(a, t, params)
    syst, lat = make_system()

    # Plot the system
    def site_col(site):
        return 'yellow' if site == lat(1) else 'black'

    def hopping_lw(site1, site2):
        return 0.1*syst[(site1, site2)]
    if plotSystem and am_master():
        kwant.plot(syst, site_color=site_col, hop_lw=hopping_lw)

    # unitE is the lead-dot hopping in energy space
    unitE = _unitE(t/params['lam'], a)
    unit_lifetime = t/params['lam']/2/unitE/unitE

    # Perform tKwant calculations
    results = calculations(syst, lat, ilead, tmax, t, boundaries_type,
                           num_cells, strength, degree, times,
                           integration_error, params)


def _unitE(t, a):
    return 0.2 * t


if __name__ == '__main__':
    # Parameters ######################################################
    ###################################################################

    # System building parameter
    a = 1.  # Lattice spacing
    t = 1 / (a ** 2)  # Hopping term
    unitE = _unitE(t, a)
    lam = 1  # rescaling parameter for wide band limit (lam \to \infty)
    t_lam = lam * t
    tc = unitE  # hopping to dot
    tc_lam = sqrt(lam) * tc
    energyOffset = 0

    # Hamiltonian parameters
    params = {
        'energyOffset': energyOffset,  # Arbitrary energyOffset
        'tc': tc_lam,  # Link potential between the dot and leads
        't0': 0.0/unitE,  # Beginning of the time-dependence
        'muL':  energyOffset,  # Chemical potential into the left lead
        'muR':  energyOffset,  # Chemical potential into the right lead
        'TL': 0.01 * unitE,  # Temperature into the left lead
        'TR': 0.01 * unitE,  # Temperature into the right lead
        'V_lead':   2 * unitE,  # Additional lead voltage
        'eps_c':   0.2 * unitE,  # Initial dot energy level
        'lam':   lam,
    }

    # Parameters for the tKwant calculation
    ilead = 0  # in which lead to calculate the operator
    # 1) Boundary conditions
    boundaries_type = ['simple_boundaries', 'absorbing_boundaries'][0]
    num_cells = 100  # Number of cells in the lead for absorbing boundaries
    strength = 20  # Strength of absorbtion in the lead for abs. boundaries
    degree = 6  # Degree of the polynomial function used for abs. boundaries

    tmax = 10/unitE  # Duration of the simulation
    dt = 0.05/unitE  # Interval between two measures of the operator

    integration_error = 1E-5  # Allowed error of the integral over the energies

    # which figures are to be shown
    plotSystem = False
    showFigure = True
    showPlots = False

    # Run the calculation
    main(a, t_lam, ilead, boundaries_type, num_cells, strength, degree, tmax,
         dt, integration_error, plotSystem, showFigure, showPlots, params)
