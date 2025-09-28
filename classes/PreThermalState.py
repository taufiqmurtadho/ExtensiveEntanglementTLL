#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 12:27:06 2025

@author: taufiqmurtadho
"""

import numpy as np
from scipy.optimize import root_scalar

"""Class for analytically calculating covariance matrix, logarithmic negativity, and mutual information
in prethermal and thermal states"""
class PreThermalState:
    def __init__(self, com_temperature, rel_temperature, tunneling_rate_J, 
                 mean_density, condensate_length, omega_perp = 2*np.pi*2000):
        """If common temperature is different from relative temperature -> prethermal"""
        """If common temperature is the same as relative temperature -> thermal"""
        
        """Class initialization"""
        #Input variables
        self.com_temperature = com_temperature
        self.rel_temperature = rel_temperature
        self.tunneling_rate_J = tunneling_rate_J
        self.mean_density = mean_density
        self.condensate_length = condensate_length
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
        self.rel_coh_length = ((self.hbar**2)*self.mean_density)/(self.atomic_mass_m*self.kb*self.rel_temperature)
        #self.restoration_length = np.sqrt(self.hbar/(4*self.atomic_mass_m*self.tunneling_rate_J))
    
    def calc_mean_occupation(self, momentum_k, temperature_T, tunneling_rate_J):
        """Function to calculate mean occupation at equilibrium for a given momentum and temperature"""
        Ek = (((self.hbar*momentum_k)**2)/(2*self.atomic_mass_m))+2*self.hbar*tunneling_rate_J
        epsilon_k = np.sqrt(Ek*(Ek+2*self.interaction_strength_g*self.mean_density))
        thermal_energy = self.kb*temperature_T
        eta_k = (np.exp(epsilon_k/thermal_energy) - 1)**(-1)
        return eta_k
        
    
    def calc_phase_phase_sector(self, momentum_cutoff):
        """Function to calculate the phase-phase sector of the covariance matrix (at equilibrium)"""
        k1 = np.pi/self.condensate_length
        momentum_k_list = [n*k1 for n in range(1,momentum_cutoff+1)]
        
        mean_occupation_plus_list = [self.calc_mean_occupation(k, self.com_temperature, 0) for k in momentum_k_list]
        mean_occupation_minus_list = [self.calc_mean_occupation(k, self.rel_temperature, self.tunneling_rate_J) for k in momentum_k_list]
        
        phase_phase_cov_mat = np.zeros((2*momentum_cutoff, 2*momentum_cutoff))
        for i in range(momentum_cutoff):
            Ek = (((self.hbar*momentum_k_list[i])**2)/(2*self.atomic_mass_m))
            plus_elem_k = ((1+(2*self.interaction_strength_g*self.mean_density/Ek))**(1/2))*(1+2*mean_occupation_plus_list[i])/(2*self.mean_density*self.healing_length)
            minus_elem_k =((1+(2*self.interaction_strength_g*self.mean_density/(Ek+2*self.hbar*self.tunneling_rate_J)))**(1/2))*(1+2*mean_occupation_minus_list[i])/(2*self.mean_density*self.healing_length)
            
            #Filling diagonal elements
            phase_phase_cov_mat[i,i] = (plus_elem_k+minus_elem_k)/4
            phase_phase_cov_mat[momentum_cutoff+i, momentum_cutoff+i] = (plus_elem_k+minus_elem_k)/4
            
            #Filling off-diagonal elements
            phase_phase_cov_mat[i, momentum_cutoff+i] = (plus_elem_k - minus_elem_k)/4
            phase_phase_cov_mat[momentum_cutoff+i, i] = (plus_elem_k - minus_elem_k)/4
        
        return phase_phase_cov_mat
    
    def calc_density_density_sector(self, momentum_cutoff):
        """Function to calculate the density-density sector of the covariance matrix (at equilibrium)"""
        k1 = np.pi/self.condensate_length
        momentum_k_list = [n*k1 for n in range(1,momentum_cutoff+1)]
        
        mean_occupation_plus_list = [self.calc_mean_occupation(k, self.com_temperature, 0) for k in momentum_k_list]
        mean_occupation_minus_list = [self.calc_mean_occupation(k, self.rel_temperature, self.tunneling_rate_J) 
                                      for k in momentum_k_list]
        
        density_density_cov_mat = np.zeros((2*momentum_cutoff, 2*momentum_cutoff))
        for i in range(momentum_cutoff):
            Ek = (((self.hbar*momentum_k_list[i])**2)/(2*self.atomic_mass_m))
            plus_elem_k = ((1+(2*self.interaction_strength_g*self.mean_density/Ek))**(-1/2))*(1+2*mean_occupation_plus_list[i])*(2*self.mean_density*self.healing_length)
            minus_elem_k =((1+(2*self.interaction_strength_g*self.mean_density/(Ek+2*self.hbar*self.tunneling_rate_J)))**(-1/2))*(1+2*mean_occupation_minus_list[i])*(2*self.mean_density*self.healing_length)
            
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
        """Function to calculate logarithmic negativity using analytical formula"""
        k1 = np.pi/self.condensate_length
        momentum_k_list = [n*k1 for n in range(1,momentum_cutoff+1)]
        
        log_neg = 0
        
        for i in range(len(momentum_k_list)):
            
            Ek = ((self.hbar*momentum_k_list[i])**2)/(2*self.atomic_mass_m)
            occupation_plus = self.calc_mean_occupation(momentum_k_list[i], self.com_temperature, 0)
            occupation_minus = self.calc_mean_occupation(momentum_k_list[i], self.rel_temperature, self.tunneling_rate_J)
            
            thermal_term = np.sqrt((1+2*occupation_plus)*(1+2*occupation_minus))
            tunneling_term = Ek*(Ek+2*self.hbar*self.tunneling_rate_J+2*self.interaction_strength_g*self.mean_density)
            tunneling_term = tunneling_term/((Ek+2*self.hbar*self.tunneling_rate_J)*(Ek+2*self.interaction_strength_g*self.mean_density))
            tunneling_term = tunneling_term**(1/4)
            
            log_neg_term = -np.log2(tunneling_term*thermal_term)
            if log_neg_term>0:
                log_neg = log_neg + log_neg_term
        
        return log_neg
    
    def mutual_information_formula(self, momentum_cutoff):
        """Function to calculate mutual information using analytical formula"""
        k1 = np.pi/self.condensate_length
        momentum_k_list = [n*k1 for n in range(1,momentum_cutoff+1)]
        
        mi = 0
        for i in range(len(momentum_k_list)):
            Ek = ((self.hbar*momentum_k_list[i])**2)/(2*self.atomic_mass_m)
            
            Ck = Ek*(Ek+2*self.hbar*self.tunneling_rate_J+2*self.interaction_strength_g*self.mean_density)
            Ck = Ck/((Ek+2*self.hbar*self.tunneling_rate_J)*(Ek+2*self.interaction_strength_g*self.mean_density))
            Ck = np.sqrt(Ck)
            occupation_plus = self.calc_mean_occupation(momentum_k_list[i], self.com_temperature, 0)
            occupation_minus = self.calc_mean_occupation(momentum_k_list[i], self.rel_temperature, self.tunneling_rate_J)
            
            lambdaK = (1+2*occupation_plus)**2 + (1+2*occupation_minus)**2
            lambdaK = lambdaK + (1+2*occupation_plus)*(1+2*occupation_minus)*(Ck+(1/Ck))
            lambdaK = np.sqrt(lambdaK)/4
            
            subsystem_entropy_term = 2*(lambdaK + 1/2)*np.log2(lambdaK +1/2) - 2*(lambdaK - 1/2)*np.log2(lambdaK - 1/2)
            
            joint_entropy_term_plus = (1+occupation_plus)*np.log2(1+occupation_plus) - occupation_plus*np.log2(occupation_plus)
            joint_entropy_term_minus = (1+occupation_minus)*np.log2(1+occupation_minus) - occupation_minus*np.log2(occupation_minus)
     
            mi = mi +subsystem_entropy_term - joint_entropy_term_plus - joint_entropy_term_minus
        
        return mi
    
    def calc_critical_temperature(self, momentum_cutoff):
        """Function to compute critical temperature for a given tunneling rate, density, and condensate length"""
        """Temperatures specified in the class definition is not used in this function"""
        k1 = np.pi/self.condensate_length
        momentum_k_list = [n*k1 for n in range(1,momentum_cutoff+1)]
        
        def fopt(x, momentum_k):
            Ek = ((self.hbar*momentum_k)**2)/(2*self.atomic_mass_m)
            bogoliubov_energy_plus = np.sqrt(Ek*(Ek+2*self.interaction_strength_g*self.mean_density))
            bogoliubov_energy_minus = np.sqrt((Ek+2*self.hbar*self.tunneling_rate_J)*(Ek+2*self.hbar*self.tunneling_rate_J+2*self.interaction_strength_g*self.mean_density))
            
            const = (bogoliubov_energy_minus/bogoliubov_energy_plus)*(Ek/(Ek+2*self.hbar*self.tunneling_rate_J))
            
            return np.tanh(x)*np.tanh(bogoliubov_energy_minus*x/bogoliubov_energy_plus) - const
        
        critical_temperatures_all = []
        for i in range(len(momentum_k_list)):
            Ek = ((self.hbar*momentum_k_list[i])**2)/(2*self.atomic_mass_m)
            bogoliubov_energy_plus = np.sqrt(Ek*(Ek+2*self.interaction_strength_g*self.mean_density))
            
            gopt = lambda x: fopt(x, momentum_k_list[i])
            sol = root_scalar(gopt, bracket=[0, 10], method='brentq')
            Tck = bogoliubov_energy_plus/(2*self.kb*sol.root)
            critical_temperatures_all.append(Tck)
        
        Tc = max(critical_temperatures_all)
        self.critical_temperatures_all_modes = critical_temperatures_all
        self.critical_temperature = Tc
        
        return Tc
            
            
            
            
        
        