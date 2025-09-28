#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 01:57:03 2025

@author: taufiqmurtadho
"""

import numpy as np

"""Class for analytically calculating covariance matrix, logarithmic negativity, and mutual information
in states directly after coherent splitting"""
class CoherentlySplittedState:
    
    def __init__(self, com_temperature, mean_density, condensate_length, squeezing_param = 1, 
                 omega_perp = 2*np.pi*2000):
        """Class initialization"""
        self.com_temperature = com_temperature
        self.mean_density = mean_density
        self.condensate_length = condensate_length
        self.squeezing_param = squeezing_param
        self.omega_perp = omega_perp
        
        #Physical constants in SI unit
        self.atomic_mass_m = 86.9091835*1.66054e-27 #mass of Rb-87 (kg)
        self.hbar = 1.054571817e-34                 #Reduced Planck constant (SI)
        self.kb = 1.380649e-23                      #Boltzmann constant (SI)
        self.scattering_length = 5.2e-9             #5.2 nm scattering length
        prefactor_g = (2+3*self.scattering_length*self.mean_density)/((1+2*self.scattering_length*self.mean_density)**(3/2))
        self.interaction_strength_g = prefactor_g*(self.hbar*self.omega_perp*self.scattering_length)
        self.healing_length  = self.hbar/np.sqrt(self.interaction_strength_g*self.atomic_mass_m*self.mean_density)
        self.com_coh_length = ((self.hbar**2)*self.mean_density)/(self.atomic_mass_m*self.kb*self.com_temperature)
        
    
    def calc_mean_occupation(self, momentum_k, temperature_T):
        """A function to calculate mean occupation at equilibrium for a given momentum and temperature"""
        Ek = (((self.hbar*momentum_k)**2)/(2*self.atomic_mass_m))
        epsilon_k = np.sqrt(Ek*(Ek+2*self.interaction_strength_g*self.mean_density))
        thermal_energy = self.kb*temperature_T
        eta_k = (np.exp(epsilon_k/thermal_energy) - 1)**(-1)
        return eta_k
        
    def calc_phase_phase_sector(self, momentum_cutoff):
         """A function to calculate the phase-phase sector of the covariance matrix"""
         k1 = np.pi/self.condensate_length
         momentum_k_list = [n*k1 for n in range(1,momentum_cutoff+1)]
         
         mean_occupation_plus_list = [self.calc_mean_occupation(k, self.com_temperature) for k in momentum_k_list]
         
         phase_phase_cov_mat = np.zeros((2*momentum_cutoff, 2*momentum_cutoff))
         for i in range(momentum_cutoff):
             Ek = (((self.hbar*momentum_k_list[i])**2)/(2*self.atomic_mass_m))
             plus_elem_k = ((1+(2*self.interaction_strength_g*self.mean_density/Ek))**(1/2))*(1+2*mean_occupation_plus_list[i])/(2*self.mean_density*self.healing_length)
             minus_elem_k = 1/(self.mean_density*self.healing_length*(self.squeezing_param**2))
             
             #Filling diagonal elements
             phase_phase_cov_mat[i,i] = (plus_elem_k+minus_elem_k)/4
             phase_phase_cov_mat[momentum_cutoff+i, momentum_cutoff+i] = (plus_elem_k+minus_elem_k)/4
             
             #Filling off-diagonal elements
             phase_phase_cov_mat[i, momentum_cutoff+i] = (plus_elem_k - minus_elem_k)/4
             phase_phase_cov_mat[momentum_cutoff+i, i] = (plus_elem_k - minus_elem_k)/4
         return phase_phase_cov_mat
     
    def calc_density_density_sector(self, momentum_cutoff):
        """A function to calculate the density-density sector of the covariance matrix (at equilibrium)"""
        k1 = np.pi/self.condensate_length
        momentum_k_list = [n*k1 for n in range(1,momentum_cutoff+1)]
        
        mean_occupation_plus_list = [self.calc_mean_occupation(k, self.com_temperature) for k in momentum_k_list]
        
        density_density_cov_mat = np.zeros((2*momentum_cutoff, 2*momentum_cutoff))
        for i in range(momentum_cutoff):
            Ek = (((self.hbar*momentum_k_list[i])**2)/(2*self.atomic_mass_m))
            plus_elem_k = ((1+(2*self.interaction_strength_g*self.mean_density/Ek))**(-1/2))*(1+2*mean_occupation_plus_list[i])*(2*self.mean_density*self.healing_length)
            minus_elem_k = (self.mean_density*self.healing_length*(self.squeezing_param**2))
            
            #Filling diagonal elements
            density_density_cov_mat[i,i] = (plus_elem_k+minus_elem_k)/4
            density_density_cov_mat[momentum_cutoff+i, momentum_cutoff+i] = (plus_elem_k+minus_elem_k)/4
            
            #Filling off-diagonal elements
            density_density_cov_mat[i, momentum_cutoff+i] = (plus_elem_k - minus_elem_k)/4
            density_density_cov_mat[momentum_cutoff+i, i] = (plus_elem_k - minus_elem_k)/4
        
        return density_density_cov_mat
    
    def calc_phase_density_sector(self, momentum_cutoff):
        """In equilibrium the phase density sector is identically zero"""
        return np.zeros((2*momentum_cutoff, 2*momentum_cutoff))
    
    def cov_subsys_corr_decomposition(self, momentum_cutoff):
        """A function that returns decomposition of cov mat into subsystem and correlation"""
        subsys_cov = np.zeros((2*momentum_cutoff, 2*momentum_cutoff))
        corr_cov = np.zeros((2*momentum_cutoff, 2*momentum_cutoff))
        for i in range(momentum_cutoff):
            phase_phase_cm = self.calc_phase_phase_sector(momentum_cutoff)
            density_density_cm = self.calc_density_density_sector(momentum_cutoff)
            
            subsys_cov[0:momentum_cutoff, 0:momentum_cutoff] = phase_phase_cm[0:momentum_cutoff, 0:momentum_cutoff]
            subsys_cov[momentum_cutoff:2*momentum_cutoff, momentum_cutoff:2*momentum_cutoff] = density_density_cm[0:momentum_cutoff, 0:momentum_cutoff]
            
            corr_cov[0:momentum_cutoff, 0:momentum_cutoff] = phase_phase_cm[0:momentum_cutoff, momentum_cutoff:2*momentum_cutoff]
            corr_cov[momentum_cutoff:2*momentum_cutoff, momentum_cutoff:2*momentum_cutoff] = density_density_cm[0:momentum_cutoff, momentum_cutoff:2*momentum_cutoff]
            
        return [subsys_cov, corr_cov]
    
    def full_cov_mat(self, momentum_cutoff):
        """A function that returns the full covariance matrix of the bipartite system"""
        subsys_cov, corr_cov = self.cov_subsys_corr_decomposition(momentum_cutoff)
        
        full_cov = np.zeros((4*momentum_cutoff, 4*momentum_cutoff))
        full_cov[0:2*momentum_cutoff, 0:2*momentum_cutoff] = subsys_cov
        full_cov[2*momentum_cutoff:4*momentum_cutoff, 2*momentum_cutoff:4*momentum_cutoff] = subsys_cov
        full_cov[0:2*momentum_cutoff, 2*momentum_cutoff:4*momentum_cutoff] = corr_cov
        full_cov[2*momentum_cutoff:4*momentum_cutoff, 0:2*momentum_cutoff] = np.transpose(corr_cov)
        
        return full_cov

    
    def log_negativity_formula(self, momentum_cutoff):
        """A function to calculate logarithmic negativity using analytical formula"""
        k1 = np.pi/self.condensate_length
        momentum_k_list = [n*k1 for n in range(1,momentum_cutoff+1)]
        
        log_neg = 0
        for i in range(momentum_cutoff):
            Ek = (((self.hbar*momentum_k_list[i])**2)/(2*self.atomic_mass_m))
            epsilon_k = np.sqrt(Ek*(Ek+2*self.interaction_strength_g*self.mean_density))
            eta_k = self.calc_mean_occupation(momentum_k_list[i], self.com_temperature)
            
            symp_eigvals_plus =  (1/self.squeezing_param)*np.sqrt((eta_k+1/2)*Ek/epsilon_k)    
            symp_eigvals_minus = (self.squeezing_param/2)*np.sqrt((eta_k+1/2)*epsilon_k/Ek)
            
            if -np.log2(2*symp_eigvals_plus)>0:
                log_neg = log_neg - np.log2(2*symp_eigvals_plus)
            if -np.log2(2*symp_eigvals_minus)>0:
                log_neg = log_neg -np.log2(2*symp_eigvals_minus)
            
        return log_neg      
    
    def critical_temperature_formula(self, momentum_cutoff):
        """A function to calcuilate the critical temperature using analytical formula"""
        k1 = np.pi/self.condensate_length
        r = self.squeezing_param
        momentum_k_list = [n*k1 for n in range(1,momentum_cutoff+1)]
        
        critical_temperatures_all = []
        for i in range(momentum_cutoff):
            Ek = (((self.hbar*momentum_k_list[i])**2)/(2*self.atomic_mass_m))
            epsilon_k = np.sqrt(Ek*(Ek+2*self.interaction_strength_g*self.mean_density))
            Fkr = (Ek/(epsilon_k*r**2))+(epsilon_k*r**2/(4*Ek))+1
            
            x = np.log(np.sqrt(Fkr/(Fkr-2)))
            
            Tck = epsilon_k/(self.kb*x)
            
            critical_temperatures_all.append(Tck)
        
        Tc = max(critical_temperatures_all)
        self.critical_temperature = Tc
        self.critical_temperature_all_modes = critical_temperatures_all
        return Tc
    
    def mutual_information_formula(self, momentum_cutoff):
        """A function to calculate mutual information using analytical formula"""
        k1 = np.pi/self.condensate_length
        r = self.squeezing_param
        momentum_k_list = [n*k1 for n in range(1,momentum_cutoff+1)]
        
        mi = 0
        symp_eigs_subsys = []
        symp_eigs_joint = []
        for i in range(momentum_cutoff):
            Ek = (((self.hbar*momentum_k_list[i])**2)/(2*self.atomic_mass_m))
            epsilon_k = np.sqrt(Ek*(Ek+2*self.interaction_strength_g*self.mean_density))
            eta_k = self.calc_mean_occupation(momentum_k_list[i], self.com_temperature)
            Ck = ((r**2)/2)*(epsilon_k/Ek)
            
            lambda_k = np.sqrt(1+((1+2*eta_k)**2)+(1+2*eta_k)*(Ck+(1/Ck)))/4
            gamma_k = (1+2*eta_k)/2
            
            symp_eigs_subsys.append(lambda_k)
            symp_eigs_joint.append(gamma_k)
        mi = 2*self.vN_entropy(symp_eigs_subsys)-self.vN_entropy(symp_eigs_joint)
        return mi            
        
    
    @staticmethod
    def vN_entropy(symp_eigvals):
        S = [(u+0.5)*np.log2(u+0.5)-(u-0.5)*np.log2(u-0.5) for u in symp_eigvals]
        S = sum(S)
        return S
        
        