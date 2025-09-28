#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 10:48:15 2025

@author: taufiqmurtadho
"""

#Directory setup and package dependencies
import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

#Import classes
from classes.CoherentlySplittedState import CoherentlySplittedState
from classes.QuantumClassicalCorrED import QuantumClassicalCorrED


"""1. Mutual information parameter scan: mean density and temperature"""
temperature_list = np.arange(5,51,1)*1e-9
condensate_length = 100e-6
healing_length = 0.5e-6
cutoff = int(condensate_length/(2*np.pi*healing_length))
mean_density_list = np.arange(50,155,2.5)*1e6
mi_scan = np.zeros((len(mean_density_list), len(temperature_list)))
critical_temp_Tc_1 = np.zeros(len(mean_density_list))
for i in range(len(mean_density_list)):
    for j in range(len(temperature_list)):
        coh_split_state = CoherentlySplittedState(temperature_list[j], mean_density_list[i], 
                                                  condensate_length)
        mi_scan[i,j] = coh_split_state.mutual_information_formula(cutoff)
    critical_temp_Tc_1[i] = coh_split_state.critical_temperature_formula(cutoff)

"""1b. Benchmarking with ED"""
mean_density =75e6
temperature_list_coarse = np.arange(5,51,5)*1e-9
mi_ED = np.zeros(len(temperature_list_coarse))
for i in range(len(temperature_list_coarse)):
    coh_split_state = CoherentlySplittedState(temperature_list_coarse[i], mean_density, 
                                              condensate_length)
    cov = coh_split_state.full_cov_mat(cutoff)
    corr = QuantumClassicalCorrED(cov, cutoff)
    
    mi_ED[i] = corr.calc_mutual_information()

"""2. Mutual information parameter scan: squeezing parameter and temperature"""
r_list = np.arange(0.1,1.52,0.02)
critical_temp_Tc_2 = np.zeros(len(r_list))
mi_scan_2 = np.zeros((len(r_list), len(temperature_list)))
for i in range(len(r_list)):
    for j in range(len(temperature_list)):
        coh_split_state = CoherentlySplittedState(temperature_list[j], mean_density, 
                        condensate_length, squeezing_param = r_list[i])
        mi_scan_2[i,j] = coh_split_state.mutual_information_formula(cutoff)
    critical_temp_Tc_2[i] = coh_split_state.critical_temperature_formula(cutoff)

"""3. Making nice plots"""
fs_labels = 14
fs_ticks = 14
Dred = np.array([200,43,80])/238
fig, axs = plt.subplots(1, 3, figsize=(10.5,2.8), constrained_layout=True)

im1 = axs[0].plot(temperature_list*1e9, mi_scan[10,:], color = 'black')
axs[0].plot(temperature_list_coarse*1e9, mi_ED, '^', color = Dred)
axs[0].tick_params(labelsize = fs_ticks)
axs[0].set_xticks([0,25,50])
axs[0].set_xlabel(r'$T\; \rm (nK)$', fontsize = fs_labels)
axs[0].set_ylabel(r'$I(A:B)$', fontsize = fs_labels)
axs[0].axvline(critical_temp_Tc_1[10]*1e9, linestyle = '--', color = 'black' )
axs[0].legend(['Analytical', 'Numerical'])

im1 = axs[1].pcolormesh(temperature_list*1e9, mean_density_list*1e-6, mi_scan, 
                        cmap='inferno', rasterized = True)
axs[1].plot(critical_temp_Tc_1*1e9, mean_density_list*1e-6, color='white', linestyle='--')
axs[1].tick_params(labelsize=fs_ticks)
axs[1].set_xlabel(r'$T\; \rm (nK)$', fontsize = fs_labels)
axs[1].set_ylabel(r'$n_{\mathcal{1D}}\; \rm (\mu m^{-1})$', fontsize = fs_labels)
axs[1].set_yticks([50,75,100,125,150])
axs[1].set_xticks([10,30,50])

im2 = axs[2].pcolormesh(temperature_list*1e9, r_list, mi_scan_2, 
                        cmap='inferno',rasterized = True)
axs[2].plot(critical_temp_Tc_2*1e9, r_list, color='white', linestyle='--')
axs[2].set_xlim([temperature_list[0]*1e9, temperature_list[-1]*1e9])
axs[2].tick_params(labelsize=fs_ticks)
axs[2].set_xlabel(r'$T\; \rm (nK)$', fontsize = fs_labels)
axs[2].set_ylabel(r'Squeezing $r$', fontsize = fs_labels)
axs[2].set_yticks([0.1,0.5,1,1.5])
axs[2].set_xticks([10,30,50])
cbar1 = fig.colorbar(im2, ax=axs[1], orientation='vertical', aspect=30)
cbar1.ax.tick_params(labelsize=fs_ticks)
cbar1.set_ticks([20, 40, 60, 80, 100])
cbar1.set_label(r'$I(A:B)$', fontsize=fs_labels)

# Create a single colorbar for both subplots
cbar2 = fig.colorbar(im1, ax=axs[2], orientation='vertical', aspect=30)
cbar2.ax.tick_params(labelsize=fs_ticks)

cbar2.set_label(r'$I(A:B)$', fontsize=fs_labels)

labels = [r'($\mathbf{a}$)', r'($\mathbf{b}$)', r'($\mathbf{c}$)']
for i, ax in enumerate(axs):
    ax.text(-0.35, 1, labels[i], transform=ax.transAxes, 
            fontsize=14, fontweight='bold', va='top', ha='left')
    
plt.savefig('figures/coh_split_mutual_info.pdf', format='pdf', dpi=1200)
