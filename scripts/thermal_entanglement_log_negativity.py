#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 20:08:14 2025

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
from scipy.stats import linregress

#Import classes
from classes.PreThermalState import PreThermalState
from classes.QuantumClassicalCorrED import QuantumClassicalCorrED


#%%
mean_density = 75e6
temperature_list = np.arange(0.1,7.1,0.1)*1e-9
J_freqs = np.arange(0,20.5,0.5)
tunneling_rate_J_list = 2*np.pi*J_freqs
condensate_length = 100e-6
healing_length = 0.5e-6
cutoff = int(condensate_length/(2*np.pi*healing_length))

#%%%
"""1. Logarithmic negativity parameter scan"""
log_neg_scan = np.zeros((len(tunneling_rate_J_list), len(temperature_list)))
for i in range(len(tunneling_rate_J_list)):
    for j in range(len(temperature_list)):
        prethermal_state = PreThermalState(temperature_list[j], temperature_list[j], 
                                           tunneling_rate_J_list[i], mean_density, 
                                           condensate_length)
        log_neg = prethermal_state.log_negativity_formula(cutoff)
        log_neg_scan[i,j] = log_neg
        


#Computing critical temperature scan excluding region with no solution
Jfreqs2 = np.arange(0.2,20.5,0.5)
Jlist2 = 2*np.pi*Jfreqs2

Tc_first_all = np.zeros(len(Jlist2))
Tc_all = np.zeros(len(Jlist2))
for i in range(len(Jlist2)):
    prethermal_state = PreThermalState(2e-9, 2e-9, Jlist2[i], mean_density, condensate_length)
    Tc_all[i] = prethermal_state.calc_critical_temperature(cutoff)
    Tc_first_all[i] = prethermal_state.critical_temperatures_all_modes[0]
    
"""2. Benchmarking analytical formula and exact diagonalization + compute
critical temperature for entanglement"""
#Benchmarking analytical formula and exact diagonalization
chosen_J_idx = 10
tunneling_rate_J = tunneling_rate_J_list[chosen_J_idx]
temperature_list_coarse = np.arange(0.5,7.5,0.5)*1e-9
log_neg_ED = np.zeros(len(temperature_list_coarse))
log_neg_benchmark = np.zeros(len(temperature_list_coarse))
for i in range(len(temperature_list_coarse)):
    prethermal_state = PreThermalState(temperature_list_coarse[i], 
                                       temperature_list_coarse[i],
                                       tunneling_rate_J, mean_density, 
                                       condensate_length)
    
    log_neg_benchmark[i] = prethermal_state.log_negativity_formula(cutoff)
    Tc = prethermal_state.calc_critical_temperature(cutoff)
    Tc_first = prethermal_state.critical_temperatures_all_modes[0]
    cov = prethermal_state.full_cov_mat(cutoff)
    corr = QuantumClassicalCorrED(cov, cutoff)
    log_neg_ED[i] = corr.calc_log_negativity()


"""3. Show linear scaling of logarithmic negativity with length"""
condensate_length_list = np.arange(50,160,10)*1e-6
cutoff_list = [int(L/(2*np.pi*healing_length)) for L in condensate_length_list]
temperature = 2e-9

log_neg_scaling = np.zeros(len(condensate_length_list))
log_neg_scaling_ED = np.zeros(len(condensate_length_list))
for i in range(len(condensate_length_list)):
    prethermal_state = PreThermalState(temperature, temperature,
                                       tunneling_rate_J, mean_density, 
                                       condensate_length_list[i])
    
    log_neg_scaling[i] = prethermal_state.log_negativity_formula(cutoff_list[i])
    cov = prethermal_state.full_cov_mat(cutoff_list[i])
    
    corr = QuantumClassicalCorrED(cov, cutoff_list[i])
    log_neg_scaling_ED[i] = corr.calc_log_negativity()

#Linear fitting
slope, intercept, r_value, p_value, std_err = linregress(condensate_length_list*1e6, log_neg_scaling)
y_fit = slope * condensate_length_list*1e6 + intercept

"""4. Show scaling of logarithmic negativity with mean density"""
mean_density_list = np.arange(50,155,5)*1e6
log_neg_density_scaling = np.zeros(len(mean_density_list))
log_neg_density_scaling_ED = np.zeros(len(mean_density_list))
for i in range(len(mean_density_list)):
    prethermal_state = PreThermalState(temperature, temperature,
                                       tunneling_rate_J, mean_density_list[i], 
                                       condensate_length)
    
    log_neg_density_scaling[i] = prethermal_state.log_negativity_formula(cutoff)
    cov = prethermal_state.full_cov_mat(cutoff)
    
    corr = QuantumClassicalCorrED(cov, cutoff)
    log_neg_density_scaling_ED[i] = corr.calc_log_negativity()

"""5. Making nice plots"""
Dred = np.array([200,43,80])/238
fig, axs = plt.subplots(2, 2, figsize=(7,5.5), constrained_layout=True)
fs_labels = 16
fs_ticks  = 14




# First subplot
im0 = axs[0,0].pcolormesh(temperature_list*1e9, J_freqs, log_neg_scan, cmap = 'inferno',
                        rasterized = True)
axs[0,0].plot(Tc_first_all*1e9, Jfreqs2,  color = 'white', linestyle = '-.')
axs[0,0].plot(Tc_all*1e9, Jfreqs2, color = 'white', linestyle = '--')
axs[0,0].set_xlim([temperature_list[0]*1e9, temperature_list[-1]*1e9])
axs[0,0].set_ylabel(r'$\tilde{J}\; \rm (Hz)$', fontsize = fs_labels)
axs[0,0].set_xlabel(r'$T\; \rm (nK)$', fontsize = fs_labels)
axs[0,0].set_xticks([1,3,5, 7])
axs[0,0].set_xticklabels([1,3,5, 7], fontsize = fs_ticks)
axs[0,0].set_yticks([0,5,10,15,20])
axs[0,0].set_yticklabels([0,5,10,15,20], fontsize = fs_ticks)

axs[0,0].text(0.4, 12, "Entangled", color='black', fontsize= fs_ticks)
axs[0,0].text(4.6, 3, "Separable", color='white', fontsize=fs_ticks)
cbar = fig.colorbar(im0, ax=axs[0,0], orientation='vertical', aspect=30)
cbar.ax.tick_params(labelsize=14)


# Second subplot
im1 = axs[0,1].plot(temperature_list*1e9, log_neg_scan[10,:], color = 'black')
axs[0,1].plot(temperature_list_coarse*1e9, log_neg_ED, '^', color = Dred)
axs[0,1].axvline(Tc*1e9, linestyle = '--',color = 'black')
axs[0,1].axvline(Tc_first*1e9, linestyle = '-.', color = 'black' )
axs[0,1].set_ylabel(r'$E_{\mathcal{N}}$', fontsize = fs_labels)
axs[0,1].set_yticks([0,4,8,12])
axs[0,1].set_xticks([0,2,4,6])
axs[0,1].set_xlabel(r'$T\; \rm (nK)$', fontsize = fs_labels)
axs[0,1].text(4.7, 1, r'$T^*$', color='black', fontsize=fs_ticks)
axs[0,1].text(3, 4, r'$T^*_{k_1}$', color='black', fontsize=fs_ticks)
axs[0,1].legend(['Analytical','Numerical'])
axs[0,1].set_yticks([0,4,8,12])
axs[0,1].set_yticklabels([0,4,8,12], fontsize = fs_ticks)
axs[0,1].set_xticks([1,3,5,7])
axs[0,1].set_xticklabels([1,3,5,7], fontsize = fs_ticks)

#im2 = axs[1,0].plot(condensate_length_list*1e6, y_fit, color = 'black')
im2 = axs[1,0].plot(condensate_length_list*1e6, log_neg_scaling, '-', markerfacecolor='none', markersize = 8, color = 'black')
axs[1,0].plot(condensate_length_list*1e6, log_neg_scaling_ED, '^', color = Dred)
axs[1,0].set_ylabel(r'$E_{\mathcal{N}}$', fontsize = fs_labels)
axs[1,0].set_xlabel(r'$L\; \rm (\mu m)$', fontsize = fs_labels)
axs[1,0].set_yticks([2,4,6,8])
axs[1,0].set_yticklabels([2,4,6,8], fontsize = fs_ticks)
axs[1,0].set_xticks([50,75,100,125,150])
axs[1,0].set_xticklabels([50,75,100,125,150], fontsize = fs_ticks)

im3 = axs[1,1].plot(mean_density_list*1e-6, log_neg_density_scaling, color = 'black')
axs[1,1].plot(mean_density_list*1e-6, log_neg_density_scaling_ED, '^', color = Dred)
axs[1,1].set_ylabel(r'$E_{\mathcal{N}}$', fontsize = fs_labels)
axs[1,1].set_xlabel(r'$n_{\rm 1D}\; \rm (\mu m^{-1})$', fontsize = fs_labels)
axs[1,1].set_yticks([4,5,6,7])
axs[1,1].set_yticklabels([4,5,6,7], fontsize = fs_ticks)
axs[1,1].set_xticks([50,75,100,125, 150])
axs[1,1].set_xticklabels([50,75,100,125,150], fontsize = fs_ticks)

labels = [r'($\mathbf{a}$)', r'($\mathbf{b}$)', r'($\mathbf{c}$)', r'($\mathbf{d}$)']
axs = axs.flatten()  # Flatten to 1D for easy iteration

for i, ax in enumerate(axs):
    ax.text(-0.25, 1, labels[i], transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='left')

#plt.savefig('figures/thermal_log_negativity.pdf', format='pdf', dpi=1200)