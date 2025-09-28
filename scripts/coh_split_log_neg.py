#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 00:16:19 2025

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


"""1. Logarithmic negativity parameter scan: mean density and temperature"""
temperature_list = np.arange(2,51,1)*1e-9
condensate_length = 100e-6
healing_length = 0.5e-6
cutoff = int(condensate_length/(2*np.pi*healing_length))
mean_density_list = np.arange(50,155,2.5)*1e6
log_neg_scan = np.zeros((len(mean_density_list), len(temperature_list)))
critical_temp_Tc_1 = np.zeros(len(mean_density_list))

for i in range(len(mean_density_list)):
    for j in range(len(temperature_list)):
        coh_split_state = CoherentlySplittedState(temperature_list[j], mean_density_list[i], 
                                                  condensate_length)
        log_neg_scan[i,j] = coh_split_state.log_negativity_formula(cutoff)
    
    critical_temp_Tc_1[i] = coh_split_state.critical_temperature_formula(cutoff)

"""1b. Benchmarking with ED"""
mean_density =75e6
temperature_list_coarse = np.arange(5,51,5)*1e-9
log_neg_ED = np.zeros(len(temperature_list_coarse))

for i in range(len(temperature_list_coarse)):
    coh_split_state = CoherentlySplittedState(temperature_list_coarse[i], mean_density, 
                                              condensate_length)
    cov = coh_split_state.full_cov_mat(cutoff)
    corr = QuantumClassicalCorrED(cov, cutoff)
    
    log_neg_ED[i] = corr.calc_log_negativity()

"""2. Logarithmic negativity parameter scan: squeezing parameter and temperature"""
r_list = np.arange(0.1,1.52,0.02)
critical_temp_Tc_2 = np.zeros(len(r_list))
log_neg_scan_2 = np.zeros((len(r_list), len(temperature_list)))
for i in range(len(r_list)):
    for j in range(len(temperature_list)):
        coh_split_state = CoherentlySplittedState(temperature_list[j], mean_density, 
                        condensate_length, squeezing_param = r_list[i])
        log_neg_scan_2[i,j] = coh_split_state.log_negativity_formula(cutoff)
    
    critical_temp_Tc_2[i] = coh_split_state.critical_temperature_formula(cutoff)

"""3. Making nice plots"""
fs_labels = 14
fs_ticks = 14
Dred = np.array([200,43,80])/238
fig, axs = plt.subplots(1, 3, figsize=(10.75,3), constrained_layout=True)

im1 = axs[0].plot(temperature_list*1e9, log_neg_scan[10,:], color = 'black')
axs[0].plot(temperature_list_coarse*1e9, log_neg_ED, '^', color = Dred)
axs[0].axvline(critical_temp_Tc_1[10]*1e9, linestyle = '--', color = 'black' )
axs[0].tick_params(labelsize = fs_ticks)
axs[0].set_xticks([0,25,50])
axs[0].set_xlabel(r'$T\; \rm (nK)$', fontsize = fs_labels)
axs[0].set_ylabel(r'$E_{\mathcal{N}}$', fontsize = fs_labels)
axs[0].legend(['Analytical', 'Numerical'])

vmin = min(log_neg_scan.min(), log_neg_scan_2.min())
vmax = max(log_neg_scan.max(), log_neg_scan_2.max())

im1 = axs[1].pcolormesh(temperature_list*1e9, mean_density_list*1e-6, log_neg_scan, 
                        cmap='inferno', vmin = vmin, vmax = vmax, rasterized = True)
axs[1].plot(critical_temp_Tc_1*1e9, mean_density_list*1e-6, color='white', linestyle='--')
axs[1].tick_params(labelsize=fs_ticks)
axs[1].set_xlabel(r'$T\; \rm (nK)$', fontsize = fs_labels)
axs[1].set_ylabel(r'$n_{\mathcal{1D}}\; \rm (\mu m^{-1})$', fontsize = fs_labels)
axs[1].set_yticks([50,75,100,125,150])
axs[1].set_xticks([10,30,50])
axs[1].text(5, 100, "Entangled", color='white', fontsize= fs_ticks)
axs[1].text(32, 75, "Separable", color='white', fontsize=fs_ticks)

im2 = axs[2].pcolormesh(temperature_list*1e9, r_list, log_neg_scan_2, 
                        cmap='inferno', vmin = vmin, vmax = vmax, rasterized = True)
axs[2].plot(critical_temp_Tc_2*1e9, r_list, color='white', linestyle='--')
axs[2].set_xlim([temperature_list[0]*1e9, temperature_list[-1]*1e9])
axs[2].tick_params(labelsize=fs_ticks)
axs[2].set_xlabel(r'$T\; \rm (nK)$', fontsize = fs_labels)
axs[2].set_ylabel(r'Squeezing $r$', fontsize = fs_labels)
axs[2].set_yticks([0.1,0.5,1,1.5])
axs[2].set_xticks([10,30,50])
axs[2].text(6, 1.2, "Entangled", color='white', fontsize= fs_ticks)
axs[2].text(30, 0.7, "Separable", color='white', fontsize=fs_ticks)

# Create a single colorbar for both subplots
cbar = fig.colorbar(im1, ax=[axs[1], axs[2]], orientation='vertical', aspect=30)
cbar.ax.tick_params(labelsize=fs_ticks)
cbar.set_ticks([0, 20, 40, 60])
cbar.set_label(r'$E_{\mathcal{N}}$', fontsize=fs_labels)

labels = [r'($\mathbf{a}$)', r'($\mathbf{b}$)', r'($\mathbf{c}$)']
for i, ax in enumerate(axs):
    ax.text(-0.35, 1, labels[i], transform=ax.transAxes, 
            fontsize=14, fontweight='bold', va='top', ha='left')
    
#plt.savefig('figures/coh_split_log_negativity.pdf', format='pdf', dpi=1200)
