#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 30 01:15:38 2025

@author: taufiqmurtadho
"""

import sys
sys.path.append('../../')
import numpy as np
import matplotlib.pyplot as plt
from classes.CoherentlySplittedState import CoherentlySplittedState
from classes.PreThermalState import PreThermalState
from classes.LuttingerEvolution import LuttingerEvolution
from classes.QuantumClassicalCorrED import QuantumClassicalCorrED

mean_density = 75e6
T_com = 100e-9
condensate_length = 44e-6
cutoff = 10

splitState = CoherentlySplittedState(T_com, mean_density, 
                                          condensate_length)

Tpreth_rel = splitState.interaction_strength_g*mean_density/(2*splitState.kb)

prethState = PreThermalState(T_com, Tpreth_rel, 0, mean_density, condensate_length)


cov_split = splitState.full_cov_mat(cutoff)
cov_preth = prethState.full_cov_mat(cutoff)

corr_split = QuantumClassicalCorrED(cov_split, cutoff)
corr_preth = QuantumClassicalCorrED(cov_preth, cutoff)

mi_split = corr_split.calc_mutual_information()
mi_preth = corr_preth.calc_mutual_information()

print(mi_split)
print(mi_preth)