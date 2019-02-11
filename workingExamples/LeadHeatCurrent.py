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
def make_system(a, t, params):
    # Create the Bravais lattice and the system
    lat = kwant.lattice.chain(a, norbs=1)
    syst = kwant.Builder()

    # Define the region of study: add the time-dependent potential
    @tkwant.time_dependent
    def qdot_hopping(site1, site2, time, params=params):
        return -params['tc'] * faraday_flux_direct(time, params)

    # Create onsite terms
    syst[(lat(0))] = params['energyOffset']
    syst[(lat(1))] = params['energyOffset'] + params['eps_c']
    syst[(lat(2))] = params['energyOffset']
    syst[(lat(-1))] = params['energyOffset']
    syst[(lat(3))] = params['energyOffset']

    # Create hopping terms
    syst[(lat(0), lat(1))] = qdot_hopping
    syst[(lat(2), lat(1))] = -params['tc']
    syst[(lat(-1), lat(0))] = -t
    syst[(lat(2), lat(3))] = -t

    # Define and attach the leads
    lead = kwant.Builder(kwant.TranslationalSymmetry((-a,)))
    lead[lat(0)] = params['energyOffset']
    lead[lat.neighbors()] = -t
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())

    return syst, lat, lead


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


# Calculation over the system #############################################
###########################################################################
def manybody(syst, times, boundaries, occupations, integration_regions,
             integration_error, operator, params):
    # Create the solver
    S = tkwant.manybody.Solver(syst, boundaries, occupations,
                               integration_regions,
                               integration_error=integration_error,
                               args=(params, ))

    # Evolve the system forward in time, and calculate the heat current
    results = []
    start = timer.clock()
    for time in times:
        results.append(S(operator, time))
        # Console output
        if am_master():
            print(str(int(timer.clock() - start)) + ' ' + 't = ' + str(time)
                  + ', current = ' + str(results[-1]))

    return results


def calculations(syst, lat, ilead, tmax, t, boundaries_type, num_cells,
                 strength, degree, times, integration_error, params):

    # Make all the tKwant calculations

    # Finalize the system
    syst = syst.finalized()

    # set variables where to calculate heat current, depending on lead
    if ilead == 0:
        mu = params['muL']
        intracell_sites = [lat(0)]
        intercell_sites = [lat(-1)]
    elif ilead == 1:
        mu = params['muR']
        intracell_sites = [lat(2)]
        intercell_sites = [lat(3)]
    else:
        raise ValueError('ilead has to be either 0 or 1')

    # Get the t-derivative of the Hamiltonian
    def tderiv_Hamil(a, b, *args, **kwargs):
        small_h = 0.01
        time = args[0]
        retfunc = syst.hamiltonian(a, b, time+small_h, *args[1:], **kwargs) - \
            syst.hamiltonian(a, b, time, *args[1:], **kwargs)
        retfunc /= small_h
        return retfunc

    # Initialize the Heat current
    operator = \
        operatorsHeatCurrent.heatCurrentWithIc(syst, mu,
                                               intracell_sites=intracell_sites,
                                               intercell_sites=intercell_sites,
                                               tderiv_Hamil=tderiv_Hamil)

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
    band_upperE = params['energyOffset'] + 2 * t
    band_lowerE = params['energyOffset'] - 2 * t
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
    syst, lat, lead = make_system(a, t, params)

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
    file_name = 'data/HeatCurrent'
    file_name += '_error=' + str(integration_error) + file_name_basis + '.dat'

    # Save the data
    if am_master():
        write_results = []
        for i, res in enumerate(results):
            write_results.append([times[i]/unit_lifetime,
                                  res*unit_lifetime/unitE])
        write_file(write_results, file_name)

    # Plot and save the figure ########################################
    ###################################################################
    if am_master():
        fig = plot_graph(write_results)
        fig.savefig('Ih_plots/' + file_name_basis + '_error='
                    + str(integration_error) + '.png', bbox_inches='tight')
        if showFigure:
            plt.show()


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
        'V_lead':  2 * unitE,  # Additional lead voltage
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
