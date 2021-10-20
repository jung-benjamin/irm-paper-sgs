#! /usr/bin/env python3

"""Implementation of the Isotope Ratio Method

This module contains an functions for an implementation of the Isotope Ratio
Method as described in the authors manuscript submitted to Science & Global
Security. 

A core approximation in this simplified IRM implementation is that a reactor
is operated for multiple cycles that are very similar and can thus be 
approximated with an average cycle, which is called an 'average batch' in the
paper. It is assumed that after one such cycle, the entire core is emptied and
the reactor is refueled with fresh fuel.

@author Benjamin Jung
"""

import pandas as pd
import numpy as np
from scipy.linalg import expm


def calc_reaction_rate(xs_data, energy_spectrum):
    """Calculate the reaction rate from the cross sections
    
    The reaction rate is calculated by integrating the
    product of cross-section and neutron flux over energy.
    Both are available on a discrete energy grid, which
    converts the integral into a sum.
    Reaction rates for all reactions in the cross-section
    DataFrame (xs_data) are calculted. All arrays need to 
    be on the same energy grid.
    
    The JANIS cross-section data is given in units barn.
    Therefore the one-group cross-section is multiplied with
    1e-24 to transform to SI units.
    
    Parameters
    ----------
    xs_data : pd.DataFrame
        Columns contain energy dependent cross section data in 
        units of barn.
    energy_spectrum : pd.DataFrame or pd.Series
        Columns contain the energy spetrum counts.
    
    Returns
    -------
    rate : pd.Series
        One reaction rate for each reactin in xs_data
    """
    df = pd.DataFrame()
    for name, column in xs_data.items():
        prod = energy_spectrum.values * column.values
        df[name] = prod
    rate = df.sum(axis=0)
    return rate * 1e-24


def isotopic_vector(matrix, t, xs, spectrum, n_0):
    """Calculate the isotopic vector evolution
    
    Uses a simplified burnup matrix to calculate
    the evolution of the isotopic vector as a 
    function of time.
    
    Parameters
    ----------
    matrix : callable
        Simplified burnup matrix
    t : float
        Time in seconds (time the reactor is operational)
    xs : pd.DataFrame
        Cross-section data for the reactions accounted
        for in the burnup matrix
    spectrum : np.ndarray or pd.DataFrame
        Neutron spectrum, needs to be on the same energy
        grid as the cross-sections
    n_0 : np.ndarray
        Isotopic vector of the element at t=0, commonly
        the natural isotopic composition.
        
    Returns
    -------
    iso_vec : np.ndarray
        Isotopic vector at time t
    """
    reac_rate = calc_reaction_rate(xs, spectrum)
    bu_matrix = matrix(*reac_rate.values)
    exp_matrix = expm(bu_matrix * t)
    iso_vec = np.dot(exp_matrix, n_0)
    return iso_vec


def plutonium_to_time(pu, flux_average, phi_0, pu_0):
    """Approximate time in units of plutonium
    
    With the assumption that plutonium-per-unit-fluence is constant for 
    an average batch of fuel (one simulation), the total plutonium 
    over several subsequent batches is related to the operating time
    of the reactor via a linear equation.
    
    Parameters
    ----------
    pu : float
        Plutonium density in g cm-3.
    flux_average : float
        Average flux in the reactor in s-1 cm-2.
    phi_0 : float
        Fluence of an average batch in cm-2.
    pu_0 : float
        Plutonium density of an average batch in g cm-3.
    
    Returns
    -------
    t : float
        Total irradiation time in s.
    """
    t = pu * phi_0 / pu_0 / flux_average
    return t


def ratio_plutonium_function(spectrum, phi_0, pu_0, cross_sections,
                             matrix, n_0, idx):
    """Calculate the isotopic vector as a function of plutonium
    
    Combine steps 1 and 2 of the irm analysis. First compute the 
    isotopic vector as a function of reactor operating time, then
    insert the approximation between longterm plutonium production.
    
    Parameters
    ----------
    spectrum : np.ndarray or pd.DataFrame
        Average neutron spectrum on the same energy grid
        as the cross_sections.
    phi_0 : float
        Fluence of an average batch in cm-2.
    pu_0 : float
        Plutonium density (g cm-3) in the fuel at the end of an
        average batch. 
    cross_sections : pd.DataFrame
        Cross-sections of the reactions accounted for in the 
        burnup matrix.
    matrix : callable
        The simplified burnup matrix for the isotopic vector
        of the indicator element.
    n_0 : np.ndarray
        The natural isotopic vector of the indicator element.
        
    Returns
    -------
    ratio : callable
    """
    flux_average = spectrum.sum()
    def ratio(pu):
        """Callable ratio function with plutonium as variable"""
        t = plutonium_to_time(pu, flux_average, phi_0, pu_0)
        iso_vec = isotopic_vector(matrix,
                                  t,
                                  cross_sections,
                                  spectrum,
                                  n_0
                                  )
        return iso_vec[idx[0]] / iso_vec[idx[1]]
    return ratio


def plutonium_solver(func, ratio, guess):
    """Solve equation for plutonium given an isotopic ratio
    
    Uses scipy.optimize.fsolve to solve the equation:
    
                Ratio(Pu) - Ratio_measured = 0.
    
    Parameters
    ----------
    func : callable 
        Function relating the isotopic ratio with the total plutonium
        production.
    ratio : float
        Measured isotopic ratio.
    guess : float
        Starting guess for the solver.
        
    Returns
    -------
    pu_solve
    """
    def solve_func(pu):
        return func(pu) - ratio
    pu_solve = fsolve(solve_func, guess, full_output=True)
    return pu_solve[0]