#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 14:35:42 2025

@author: taufiqmurtadho
"""

#Directory setup and package dependencies
import sys
sys.path.append('../../')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
from scipy.stats import linregress

#Import classes
from classes.PreThermalState import PreThermalState
from classes.QuantumClassicalCorrED import QuantumClassicalCorrED


#%%
mean_density = 75e6
com_temperature = 5e-9
rel_temperature = 1e-9
J = 2*np.pi*5
condensate_length = 100e-6
healing_length = 0.5e-6
cutoff = int(condensate_length/(2*np.pi*healing_length))

#%%%
prethermal_state = PreThermalState(com_temperature, rel_temperature, J, mean_density, condensate_length)


Tc = prethermal_state.calc_critical_temperature(cutoff)
log_neg = prethermal_state.log_negativity_formula(cutoff)
print(log_neg)