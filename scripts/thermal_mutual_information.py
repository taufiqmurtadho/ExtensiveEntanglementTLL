#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 23:12:34 2025

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
from classes.PreThermalState import PreThermalState
from classes.QuantumClassicalCorrED import QuantumClassicalCorrED

"""1. Quantum to classical transition in the threshold temperature"""
mean_density = 75e6
temperature_list = np.arange(0.05,5.1,0.05)*1e-9
temperature_list_coarse = np.arange(0.05, 6, 1)*1e-9
tunneling_rate_J = 2*np.pi*5
condensate_length = 100e-6
healing_length = 0.5e-6
cutoff = int(condensate_length/(2*np.pi*healing_length))

log_negs = np.zeros(len(temperature_list))
mutual_infos = np.zeros(len(temperature_list))
#Computing with analytical formula
for i in range(len(temperature_list)):
    prethermal_state = PreThermalState(temperature_list[i], 
                                       temperature_list[i], 
                                       tunneling_rate_J, 
                                       mean_density, 
                                       condensate_length)
    
    log_negs[i] = prethermal_state.log_negativity_formula(cutoff)
    mutual_infos[i] = prethermal_state.mutual_information_formula(cutoff)

Tc = prethermal_state.calc_critical_temperature(cutoff)
#Benchmarking with exact diagonalization
mi_ed = np.zeros(len(temperature_list_coarse))
for i in range(len(temperature_list_coarse)):
    prethermal_state = PreThermalState(temperature_list_coarse[i], temperature_list_coarse[i], 
                                       tunneling_rate_J, mean_density, condensate_length)
    
    cov = prethermal_state.full_cov_mat(cutoff)
    corr = QuantumClassicalCorrED(cov, cutoff)
    
    mi = corr.calc_mutual_information()
    mi_ed[i] = mi

"""2. Parameter scan of mutual information"""
temperature_scan_list = np.arange(2,82,2)
J_freqs = np.arange(0.1,20.5,0.5)
tunneling_rate_J_list = 2*np.pi*J_freqs
critical_temp = np.zeros(len(J_freqs))
mutual_info_scan = np.zeros((len(tunneling_rate_J_list), len(temperature_scan_list)))

for i in range(len(tunneling_rate_J_list)):
    for j in range(len(temperature_scan_list)):
        prethermal_state = PreThermalState(temperature_scan_list[j], 
                                           temperature_scan_list[j],
                                           tunneling_rate_J_list[i], 
                                           mean_density, condensate_length)
        critical_temp[i] = prethermal_state.calc_critical_temperature(cutoff)
        
        mi = prethermal_state.mutual_information_formula(cutoff)
        mutual_info_scan[i,j] = mi

"""3. Linear scaling of mutual information with length"""
condensate_length_list = np.arange(50,160,10)*1e-6
cutoff_list = [int(L/(2*np.pi*healing_length)) for L in condensate_length_list]
temperature = 30e-9
mean_density_2 = 150e6

mi_scaling = np.zeros(len(condensate_length_list))
mi_scaling_ED = np.zeros(len(condensate_length_list))
mi_scaling_ED_2 = np.zeros(len(condensate_length_list))
for i in range(len(condensate_length_list)):
    prethermal_state = PreThermalState(temperature, temperature,
                                       tunneling_rate_J, mean_density, 
                                       condensate_length_list[i])
    prethermal_state_2 = PreThermalState(temperature, temperature,
                                       tunneling_rate_J, mean_density_2, 
                                       condensate_length_list[i])
    mi_scaling[i] = prethermal_state.mutual_information_formula(cutoff_list[i])
    cov = prethermal_state.full_cov_mat(cutoff_list[i])
    cov_2 = prethermal_state_2.full_cov_mat(cutoff_list[i])
    corr = QuantumClassicalCorrED(cov, cutoff_list[i])
    corr_2 = QuantumClassicalCorrED(cov_2, cutoff_list[i])
    mi_scaling_ED[i] = corr.calc_mutual_information()
    mi_scaling_ED_2[i] = corr_2.calc_mutual_information()
    


"""5. Make some nice plots"""
Dred = np.array([200,43,80])/238
fig, axs = plt.subplots(1, 3, figsize=(10,3), constrained_layout=True)

im0 = axs[0].plot(temperature_list*1e9, mutual_infos, color = 'black')
axs[0].plot(temperature_list_coarse*1e9, mi_ed, '^', color = Dred)
axs[0].plot(temperature_list*1e9, log_negs, ':', color = 'gray', linewidth = 2)
axs[0].axvline(Tc*1e9, linestyle = '--', color = 'gray')
axs[0].set_xticks([0,1,2,3,4,5])
axs[0].set_xlabel(r'$T\; \rm (nK)$', fontsize = 14)
axs[0].set_ylabel(r'$I(A:B)$', fontsize = 14)
axs[0].set_yticks([0,4,8,12])
axs[0].text(3.7, 4, r'$T^*$', color='black', fontsize=14)

im1 = axs[1].pcolormesh(temperature_scan_list, J_freqs, mutual_info_scan, 
                        cmap = 'inferno', rasterized = True)
axs[1].plot(critical_temp*1e9, J_freqs, color = 'black', linestyle = '--')
cbar = fig.colorbar(im1, ax=axs[1], orientation='vertical', aspect=30)
cbar.ax.tick_params(labelsize=14)
axs[1].set_ylabel(r'$\tilde{J}\; \rm (Hz)$', fontsize = 14)
axs[1].set_xlabel(r'$T\; \rm (nK)$', fontsize = 14)
axs[1].set_xticks([5,20,40,60,80])
im2 = axs[2].plot(condensate_length_list*1e6,mi_scaling, color = 'black')
axs[2].plot(condensate_length_list*1e6,mi_scaling_ED, '^', color = Dred, markersize = 11)
axs[2].plot(condensate_length_list*1e6, mi_scaling_ED_2, 'o', markerfacecolor = 'white', markeredgecolor = 'black', markersize = 6)
axs[2].set_ylabel(r'$I(A:B)$', fontsize = 14)
axs[2].set_xlabel(r'$L\; \rm (\mu m)$', fontsize = 14)
axs[2].legend(['Analytical', r'Num. $n_{\rm 1D}= 75\; \rm \mu m^{-1}$',
               r'Num. $n_{\rm 1D}= 150\; \rm \mu m^{-1}$'], fontsize = 10)
axs[2].set_ylim([3,18])

for ax in axs:
    ax.tick_params(axis='both', which='major', labelsize=14)

labels = [r'($\mathbf{a}$)', r'($\mathbf{b}$)', r'($\mathbf{c}$)']
for i, ax in enumerate(axs):
    ax.text(-0.25, 1, labels[i], transform=ax.transAxes, 
            fontsize=14, fontweight='bold', va='top', ha='left')

#plt.savefig('figures/thermal_mutual_information.pdf', format='pdf', dpi=1200)
