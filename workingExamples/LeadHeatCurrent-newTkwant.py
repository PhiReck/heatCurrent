# Quantum dot with variaying potential
# Model : infinite and discretized line at a commun potential, which comprizes one element at a higher and varying potential, in order to model
# a dirac potential
# The Fermi-Dirac distribution is used into the leads
# The calculation is performed without the manybody solver, using the one_body one

import numpy as np
from math import erf, sqrt
import cmath
import matplotlib.pyplot as plt
import scipy as sp
import kwant
import tkwant
#import operatorsHeatCurrent
import collections
from mpi4py import MPI

import time as timer

import os

directories = ['IvsE', 'Ivsk', 'JvsE', 'Jvsk', 'vvsE', 'vvsk', 'occupationvsE', 'occupationvsk', 'JtimesvvsE', 'Jtimesvvsk', 'Evsk', 'data', 'CurrentvsTime']
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)

###XXX ATM, ilead has to be zero (i.e. left lead!!)
ilead = 0

##### Create the system ###################################################
###########################################################################

def am_master():
    return MPI.COMM_WORLD.rank == 0

def make_system(a, t, params):
    # Create the Bravais lattice and the system
    lat = kwant.lattice.chain(a, norbs = 1)
    syst = kwant.Builder()

    # Define the region of study : add the time-dependent potential
    @tkwant.time_dependent
    def qdot_hopping(site1, site2, time, params = params): return -params['tc'] * faraday_flux_direct(time, params)

    syst[(lat(0))] = params['energyOffset']
    syst[(lat(1))] = params['energyOffset'] + params['eps_c']
    syst[(lat(2))] = params['energyOffset']
    # syst[lat.neighbors()] = +params['tc']
    syst[(lat(0),lat(1))] = qdot_hopping
    syst[(lat(2),lat(1))] = -params['tc']

#    syst[(lat(-3))] = params['energyOffset']
#    syst[(lat(-2))] = params['energyOffset']
#    syst[(lat(-1))] = params['energyOffset']
#    syst[(lat(-1),lat(0))] = -t
#    syst[(lat(-1),lat(-2))] = -t
#    syst[(lat(-3),lat(-2))] = -t


    # Define and attach the leads
    lead = kwant.Builder(kwant.TranslationalSymmetry((-a,)))
    lead[lat(0)] = params['energyOffset']
    lead[lat.neighbors()] = -t
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())

    return syst, lat, lead


##### Varying potential of the quantum dot ################################
###########################################################################

def V_erf(time, params):
    # For adding a time-dependent potential on top of the quantum dot
    # Shape : erf
    assert(params['t0'] - 3 * params['sigma'] > 0), "Potential is not null at t = 0 or is shifted at negative times"
    retour = params['V'] * params['sigma']  * np.sqrt(np.pi / 2) * (1 + erf((time - params['t0']) / (params['sigma'] * np.sqrt(2))))
    val0 = params['V'] * params['sigma']  * np.sqrt(np.pi / 2) * (1 + erf((0 - params['t0']) / (params['sigma'] * np.sqrt(2))))
    if retour > val0:
        return retour - val0
    else:
        return 0.

# def V_step(time, params):
#    # For adding a time-dependent potential on top of the quantum dot
#    # Shape : step
#    if time > params['t0']:
#        return params['eps0']+params['gamma0']
#    else:
#        return params['eps0']

def zerofct(time, *args, **kwargs):
    return 0


def faraday_flux_direct(time, *args, **kwargs):
#    omega = 0.1
    t_start = args[0]['t0']
    if time <= t_start:
        return 1
    #    elif 0 < time < t_upper:
    #        return V_dynamic * (time - sin(omega * time) / omega) / 2
    else:
        return cmath.exp(1j*args[0]['V_lead'] * (time - t_start) )


def faraday_flux(time, *args, **kwargs):
#    omega = 0.1
    # print('args:',args)
    t_start = args[0]['t0']
    if time <= t_start:
        return 0
#    elif 0 < time < t_upper:
#        return V_dynamic * (time - sin(omega * time) / omega) / 2
    else:
        return args[0]['V_lead'] * (time - t_start)


##### Useful functions ####################################################
###########################################################################

def write_file(array, file_name):
    # To write data (a 2D array) into a file in a manner that is understandable by np.genfromdat
    data = open(file_name, 'w')
    for j in array:
        for i in j:
            data.write(str(i) + ' ')
        data.write('\n')
    data.close()


def plot_graph(times, integration_methods, file_name_basis):
    # To plot the figure as a graph
    # Opens the file named "file_name_basis + '.dat'"

    fig, ax = plt.subplots(nrows = 1, ncols = 1) # Create the subplot

    for method in integration_methods:
        file_name = 'data/HeatCurrent_lead'  + str(ilead) + "_" + str(method).split(' ')[1]
        if method == manybody: file_name += '_error=' + str(integration_error)
        if method == onebody_E or method== onebody_k: file_name += '_nbpoints=' + str(nb_points)
        file_name += file_name_basis + '.dat'

        # Retrieve the data from the file
        results = np.genfromtxt(file_name)

        for i in [1]:
            curve = np.array([])
            for time in range(len(results)):
                curve = np.append(curve, results[time, i])

            ax.plot(times, curve, label = str(method).split(' ')[1] + ' lead ' + str(i))

    ax.set_title('Quantum dot : heatcurrent vs time')
    ax.set_ylabel('I_h tau/V')
    ax.set_xlabel('t  / tau')
#    ax.set_ylabel(quant1_title)
#    ax.set_xlabel(quant2_title)
    ax.legend(loc = 'lower right', fontsize = 'small')

    return fig


def plot_for_lead(lead, energies, momenta, currents, occupation, velocities, results_lead, tmax, dt, times, hoppings_map, plots, plot_times, plot_hoppings, file_name_basis, showPlots, params):
    # Plot quantities for a single lead
    # 'plots' contains a list of the plots to be made amongst ['IvsE', 'Ivsk', 'JvsE', 'Jvsk', 'vvsE', 'vvsk', 'occupationvsE', 'occupationvsk', 'JtimesvvsE', 'Jtimesvvsk', 'Evsk']
    # 'plot_times' is a list of the times at which the plots will be made
    # 'plot_hoppings' is a list of the hoppings at which the plots will be made
    assert(max(plot_times) <= tmax), "Please define correct times to be plotted"

    global quant1_title
    global quant2_title

    for plot in plots:
        fig, ax = plt.subplots(nrows = 1, ncols = 1) # Create the subplot

        if plot.split('vs')[0] == 'I':
            quant1 = results_lead
            quant1_title = 'particle current'
            multi_times = True
        elif plot.split('vs')[0] == 'J':
            quant1 = currents
            quant1_title = 'probability current'
            multi_times = True
        elif plot.split('vs')[0] == 'v':
            quant1 = velocities
            quant1_title = 'velocity'
            multi_times = False
        elif plot.split('vs')[0] == 'occupation':
            quant1 = occupation
            quant1_title = 'Fermi-Dirac distribution'
            multi_times = False
        elif plot.split('vs')[0] == 'Jtimesv':
            quant1 = currents * np.array([velocity * np.ones((len(times), len(hoppings_map))) for velocity in velocities])
            quant1_title = 'probability current times velocity'
            multi_times = True
        elif plot.split('vs')[0] == 'E':
            quant1 = energies
            quant1_title = 'energy'
            multi_times = False
        else:
            raise ValueError("Ill-defined first quantity for plot : " + plot)
        if plot.split('vs')[1] == 'E':
            quant2 = energies
            quant2_title = 'energy'
        elif plot.split('vs')[1] == 'k':
            quant2 = momenta
            quant2_title = 'momentum'
        else:
            raise ValueError("Ill-defined second quantity for plot : " + plot)

        quant1 = np.transpose(quant1)

        if multi_times:
            for time in plot_times:
                for hop in plot_hoppings:
                    ax.plot(quant2, quant1[hop][int(time / dt)], label = 'time=' + str(time) + ' hop=' + str(hop))
            ax.legend(loc = 'lower right', fontsize = 'small')
        else:
            ax.plot(quant2, quant1)

        ax.set_title('Quantum dot : ' + quant1_title + ' vs ' + quant2_title + ' for lead ' + str(lead))
        ax.set_ylabel('Current')
        ax.set_xlabel('Momentum')
        fig.savefig(plot + '/' + plot + 'lead=' + str(lead) + file_name_basis + '.png')
        if showPlots:
            fig.show()
        else:
            plt.close()


##### Calculation over the system #########################################
###########################################################################

def manybody(syst, lead_finalized, times, hoppings_map, boundaries, occupations, integration_regions, integration_error, operator, nb_points, plots, plot_times, plot_hoppings, file_name_basis, showPlots, params):
    # Create the solver
    
    # Occupation, for each lead
    occupation = [None] * len(syst.leads)
    occupation[0] = tkwant.manybody.occupation(chemical_potential=params['muL'], temperature=params['TL'])
    occupation[1] = tkwant.manybody.occupation(chemical_potential=params['muR'], temperature=params['TR'])
#    S = tkwant.manybody.Solver(syst, boundaries, occupations, integration_regions, integration_error = integration_error, args = (params, ))
    solver = tkwant.manybody.Solver(syst, occupation, times[-1], args = (params, ))

    # Have the system evolve forward in time, calculating the operator over the system
    results = []
    start = timer.clock()
    for time in times:
        solver.evolve(time)        
        error = solver.estimate_error()
        if am_master():
            print('estimated integration error= {}'.format(error))
        if error > 1e-5:
            solver.refine_intervals()
        results.append(solver.evaluate(operator))
#        results.append(S(operator, time))
        if am_master():
            print(str(int(timer.clock() - start)) + ' ' + 't = ' + str(time) + ', current = ' + str(results[-1]))

    return results


def onebody(syst, lead_finalized, times, hoppings_map, boundaries, occupations, integration_regions, operator, nb_points, variable, plots, plot_times, plot_hoppings, file_name_basis, showPlots, params):
    # Make the calculation of the operator over the energies and the lead (without adapting the energies) in order to integrate with a simpson method
    results = np.zeros((len(hoppings_map), len(times)))

    leads = [0, 1]

    start = timer.clock()

    for lead in leads:
        # Define the energies over which the operator will be calculated
        energies = np.linspace(integration_regions[lead][2][0], integration_regions[lead][2][1], nb_points)

        currents = np.zeros((len(energies), len(times), len(hoppings_map)))
        occupation = np.zeros(len(energies))
        momenta = np.zeros(len(energies))
        velocities = np.zeros(len(energies))

        for i, energy in enumerate(energies):
            # Create initial scattering state
            scattering_states = kwant.wave_function(syst, energy = energy, args = (0, params))
            psi_st = scattering_states(lead)[0]

            # Create time-dependent wave functions that start in a scattering state
            solver = tkwant.onebody.solvers.default
            psi = tkwant.solve(syst, boundaries = boundaries, psi_init = psi_st, energy = energy, solver_type=solver)

            # Have the system evolve forward in time, calculating the operator over the system
            occupation[i] = occupations[lead](energy)
            currents[i] = [operator(psi(time), args = (time, params)) for time in times]

            # Calculate momenta and velocities of the scattering state associated with 'energy'
            modes, other = lead_finalized.modes(energy, args = (0, params))
            momenta[i] = modes.momenta[1] # '1' stands for propagating states ('0' or '-1' for contra-propagating states I guess)
            velocities[i] = modes.velocities[1]

            if am_master():
                print(str(int(timer.clock() - start)) + ' energy = ' + str(energy))

        occupation_extended = np.array([oc * np.ones((len(times), len(hoppings_map))) for oc in occupation])
        velocities_extended = np.array([velocity * np.ones((len(times), len(hoppings_map))) for velocity in velocities])
        if variable == 'E':
            results_lead = currents * occupation_extended / (2 * np.pi)
            integration_variable = energies
        elif variable == 'k':
            results_lead = currents * velocities_extended * occupation_extended / (2 * np.pi)
            integration_variable = momenta
        else:
            raise ValueError("Ill-defined integration variable")

        # Plot quantities like current or velocities as functions of energies or momenta, for each lead
        plot_for_lead(lead, energies, momenta, currents, occupation, velocities, results_lead, tmax, dt, times, hoppings_map, plots, plot_times, plot_hoppings, file_name_basis, showPlots, params)

        # In order to integrate over a line and not over a column : each line becomes a single time, each column becomes a single energy/momentum
        results_lead = np.transpose(results_lead)

        # Integrate with a simpson method over the energies
        results = results + np.array([np.array([sp.integrate.simps(results_time, integration_variable) for results_time in results_hopping]) for results_hopping in results_lead])

    # Transpose for the results to have the same format as the one produced by manybody.Solver.solve()
    results = np.transpose(results)

    return results


def onebody_E(syst, lead, times, hoppings_map, boundaries, occupations, integration_regions, integration_error, operator, nb_points, plots, plot_times, plot_hoppings, file_name_basis, showPlots, params):
    return onebody(syst, lead.finalized(), times, hoppings_map, boundaries, occupations, integration_regions, operator, nb_points, 'E', plots, plot_times, plot_hoppings, file_name_basis, showPlots, params)


def onebody_k(syst, lead, times, hoppings_map, boundaries, occupations, integration_regions, integration_error, operator, nb_points, plots, plot_times, plot_hoppings, file_name_basis, showPlots, params):
    return onebody(syst, lead.finalized(), times, hoppings_map, boundaries, occupations, integration_regions, operator, nb_points, 'k', plots, plot_times, plot_hoppings, file_name_basis, showPlots, params)


def calculations(syst, lat, lead, tmax, t, boundaries_type, num_cells, strength, degree, times, method, nb_points, integration_error, plots, plot_times, plot_hoppings, file_name_basis, showPlots, params):

    # Make all the tKwant calculations

    leadm0 = tkwant.leads.add_voltage(syst, ilead, zerofct)
    # leadm1 = tkwant.leads.add_voltage(syst, ilead, zerofct)
    # leadm2 = tkwant.leads.add_voltage(syst, ilead, zerofct)

    # Finalize the system
    syst = syst.finalized()

    # Create an observable for calculating a property of the system
    hoppings_map = [(lat(1), lat(0))]#[(leadm0[0], leadm1[0])]#
    #[(lat(1), lat(0)), (lat(1), lat(2))]
    #[(lat(-2), lat(-3)),(lat(-1), lat(-2)),(lat(0), lat(-1)),(lat(1), lat(0)), (lat(1), lat(2))]
    # operator = kwant.operator.Current(syst, where = hoppings_map, sum=True)

    if ilead == 0:
        mu = params['muL']
    elif ilead == 1:
        mu = params['muR']
    else:
        raise ValueError('ilead has to be either 0 or 1')

    def tderiv_Hamil(a,b,*args, **kwargs):
        small_h = 0.01
        time = args[0]
        # retfunc = syst.hamiltonian(a, b, *args, **kwargs)
        retfunc = (syst.hamiltonian(a, b, time+small_h, *args[1:], **kwargs) - syst.hamiltonian(a, b, time, *args[1:], **kwargs))/small_h
        return retfunc

    operator = tkwant.operatorsHeatCurrent.heatCurrentWithIc(syst, mu,
                            intracell_sites=[lat(0)], intercell_sites=leadm0,
                            tderiv_Hamil=tderiv_Hamil)
    # operator = operatorsHeatCurrent.heatCurrent(syst, mu, intracell_sites = [lat(0)], intercell_sites = leadm0)

    # Boundary conditions
    if boundaries_type == 'simple_boundaries':
        boundaries = [tkwant.leads.SimpleBoundary(max_time = tmax) for lead in syst.leads]
    elif boundaries_type == 'absorbing_boundaries':
        boundaries = [tkwant.leads.MonomialAbsorbingBoundary(num_cells = num_cells, strength = strength, degree = degree) for l in syst.leads]
    else:
        raise ValueError("Ill-defined boundary conditions in the leads")
    band_upperE = params['energyOffset'] + 2 * t
    band_lowerE = params['energyOffset'] - 2 * t
    integration_regions = [(0, 0, (1e-6 + band_lowerE, band_upperE - 1e-6)),
                               (1, 0, (1e-6 + band_lowerE, band_upperE - 1e-6))]

    # Occupation, for each lead
    occupations = 0# [tkwant.manybody.fermi_dirac(params['muL'], params['TL']),
                    #   tkwant.manybody.fermi_dirac(params['muR'], params['TR'])]

    results = method(syst, lead, times, hoppings_map, boundaries, occupations, integration_regions, integration_error, operator, nb_points, plots, plot_times, plot_hoppings, file_name_basis, showPlots, params)

    return results


##### Main ################################################################
###########################################################################

def main(a, t, boundaries_type, num_cells, strength, degree, tmax, dt, integration_methods, integration_error, nb_points, plotSystem, showFigure, run_simulation, plots, plot_times, plot_hoppings, showPlots, params):
    # File name basis : used to write and read the data files and write the figure files
#    file_name_basis = '_'
#    od = collections.OrderedDict(sorted(params.items()))
#    for key, val in od.items():
#        file_name_basis += key + str(val)
    file_name_basis = '_tc=' + str("{:.3f}".format(params['tc'])) + \
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
#                      '_eps0=' + str("{:.3f}".format(params['eps0'])) + \
#                      '_gamma0=' + str("{:.3f}".format(params['gamma0']))

    times = np.arange(0, tmax, dt)

    ##### Simulation ##################################################
    ###################################################################

    # Create the initial quantum dot system
    syst, lat, lead = make_system(a, t, params)

    # Verify the system
    def site_col(site):
        return 'yellow' if site == lat(1) else 'black'
    def hopping_lw(site1,site2):
        return 0.1*syst[(site1,site2)]
    if plotSystem and am_master():
        kwant.plot(syst, site_color=site_col, hop_lw=hopping_lw)


#    tkwant.leads.add_voltage(syst, 0, zerofct)
    unitE = _unitE(t/params['lam'],a)  #unitE is the lead-dot hopping in energy space
    unit_lifetime = t/params['lam']/2/unitE/unitE
    # unit_lifetime = t/2/params['tc']/params['tc']
    # Perform tKwant calculations
    for i, method in enumerate(integration_methods):
        if run_simulation[i]:
            results = calculations(syst, lat, lead, tmax, t, boundaries_type, num_cells, strength, degree, times, method, nb_points, integration_error, plots, plot_times, plot_hoppings, file_name_basis, showPlots, params)
            file_name = 'data/HeatCurrent_lead'  + str(ilead) + "_" + str(method).split(' ')[1]
            if method == manybody: file_name += '_error=' + str(integration_error)
            if method == onebody_E or method == onebody_k: file_name += '_nbpoints=' + str(nb_points)
            file_name += file_name_basis + '.dat'
            if am_master():
                write_results = []
                for i,res in enumerate(results):
                    write_results.append([times[i]/unit_lifetime,res[0]*unit_lifetime/unitE])  #,res[1]*unit_lifetime])
                write_file(write_results, file_name)

    ##### Plot the figure #############################################
    ###################################################################
    if am_master():
        fig = plot_graph(times*unitE, integration_methods, file_name_basis)

        fig.savefig('CurrentvsTime/' + file_name_basis + '_error=' + str(integration_error) + '_nbpoints=' + str(nb_points) + '.png', bbox_inches = 'tight')
        if showFigure:
            plt.show()

def _unitE(t,a):
    return 0.2 * t

if __name__ == '__main__':
    ##### Parameters ##################################################
    ###################################################################

    # System building parameter
    a = 1. # Lattice spacing
    t = 1 / (a ** 2) # Hopping term
    tc = 0.2 * t
#    gammaWBL =
    for lam in [1]:#lam = 1
        t_lam = lam * t
        unitE = _unitE(t,a)
        tc = unitE#/sqrt(2)
        tc_lam = sqrt(lam) * tc
        energyOffset = 0  #1e-10 * t

        # Hamiltonian parameters
        params = {
            'energyOffset': energyOffset, #energyOffset such that bands start at E=0
            'tc': tc_lam, # Link potentiel between the site with the dirac potential and the surrounding sites : defines the quantum dot
    #        'V': 0.1 * t, # Maximum height of the dirac potential
            't0': 0.0/unitE, # Beginning of the pulse : pulse * heaviside(t - t0) ; or center of the erf pulse
    #        'sigma': 5, # Typical width of the (erf) pulse. Must be < t0 / 3
            'muL':  energyOffset, # Chemical potential into the left lead
            'muR':  energyOffset, # Chemical potential into the right lead
            'TL': 0.01 * unitE, # Temperature into the left lead
            'TR': 0.01 * unitE, # Temperature into the right lead
            'V_lead':   2 * unitE, # additional lead voltage
            'eps_c':   0.2 * unitE, # initial dot energy level
            'lam':   lam,
    #        'eps0':   0.5 * unitE, # initial dot energy level
    #        'gamma0':  2.5 * unitE # change of dot energy level
        }

        # Parameters for the tKwant calculation
        # 1) Boundary conditions
        boundaries_type = ['simple_boundaries', 'absorbing_boundaries'][0]
        num_cells = 100 # Number of cells in the lead in case of absorbing boundaries
        strength = 20 # Strength of absorbtion into the lead in case of absorbing boundaries
        degree = 6 # Degree of the polynomial function that is used in case of absorbing boundaries
        # 2) Other parameters
        tmax = 10/unitE # Duration of the simulation
        dt = 0.05/unitE # Interval between two measures of the operator over the system
        integration_methods = [manybody]#, onebody_E, onebody_k] # Calculations are redundant between onebody_E and onebody_k, but it would be cumbersome to define it better
        run_simulation = [True]#, False, False] # If you just read the files of a previous simulation, you don't need to run the simulation again. Must have same dimension as 'integration_methods'
        integration_error = 1E-5 # Integration error during the tKwant calculation of the integral over the energies
        nb_points = 101 # Number of regularly spaced points used to perform the calculation of the integral over the energy

        # Parameters of the plots
        plotSystem = False
        showFigure = True
        showPlots = False
        plots = ['IvsE', 'Ivsk', 'JvsE', 'Jvsk', 'vvsE', 'vvsk', 'occupationvsE', 'occupationvsk', 'JtimesvvsE', 'Jtimesvvsk', 'Evsk']
    #    plots = ['IvsE', 'Ivsk', 'JvsE', 'Jvsk', 'vvsE', 'vvsk', 'occupationvsE', 'occupationvsk', 'JtimesvvsE', 'Jtimesvvsk', 'Evsk']
        plot_times = [0, 1, 2, 10, 20, 30, 40]
        plot_hoppings = [0]   # [0] for current between 0 and 1 and [1] for current between 1 and 2 (or [0,1] for both)

        # Run the calculation
        main(a, t_lam, boundaries_type, num_cells, strength, degree, tmax, dt, integration_methods, integration_error, nb_points, plotSystem, showFigure, run_simulation, plots, plot_times, plot_hoppings, showPlots, params)
