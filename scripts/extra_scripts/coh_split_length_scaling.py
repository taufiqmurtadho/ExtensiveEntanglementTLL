#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 05:03:43 2025

@author: taufiqmurtadho
"""

import sys
sys.path.append('../../')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

#Import classes
from classes.CoherentlySplittedState import CoherentlySplittedState
#from classes.QuantumClassicalCorrED import QuantumClassicalCorrED


"""1. Logarithmic negativity and mutual info parameter length scaling"""
"""
temperature = 10e-9

condensate_length_list = np.arange(50,160,5)*1e-6
healing_length = 0.5e-6
cutoff_list = [int(L/(2*np.pi*healing_length)) for L in condensate_length_list]
mean_density = 75e6
log_neg_scan = np.zeros(len(condensate_length_list))

for i in range(len(condensate_length_list)):
    coh_split_state = CoherentlySplittedState(temperature, mean_density, 
                                                  condensate_length_list[i])
    log_neg_scan[i] = coh_split_state.log_negativity_formula(cutoff_list[i])

plt.plot(log_neg_scan, 'o')
"""

temperature = 50e-9

condensate_length_list = np.arange(50,160,5)*1e-6
healing_length = 0.5e-6
cutoff_list = [int(L/(2*np.pi*healing_length)) for L in condensate_length_list]
mean_density = 75e6
mi_scan = np.zeros(len(condensate_length_list))

for i in range(len(condensate_length_list)):
    coh_split_state = CoherentlySplittedState(temperature, mean_density, 
                                                  condensate_length_list[i])
    mi_scan[i] = coh_split_state.mutual_information_formula(cutoff_list[i])

plt.plot(mi_scan, 'o')