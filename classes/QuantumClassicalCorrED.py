#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 20:31:46 2025

@author: taufiqmurtadho
"""

import numpy as np
from numpy.linalg import eigvals
from scipy.linalg import block_diag
 #%%

"""Class for calculating quantum and classical information quantities from covariance matrix
by exact diagonalization (ED)"""
class QuantumClassicalCorrED:
    
    """Class initialization"""
    """We assume bipartite covariance matrix of size 4N x 4N with N being the fourier cutoff"""
    def __init__(self, bipartite_cov_mat, fourier_cutoff):
        self.cov_mat = bipartite_cov_mat
        self.fourier_cutoff = fourier_cutoff
        
        # Run physicality check during initialization
        eigenvalues = self.check_heisenberg_uncertainty()
        negative_eigs = eigenvalues[eigenvalues < 0]
        if negative_eigs.size > 0:
            print("Warning: Negative eigenvalues of Γ + iΩ/2:")
            print(negative_eigs)
    
    """Checking if the covariance matrix satisfies Heisenberg uncertainty relation"""
    def check_heisenberg_uncertainty(self, cov = None):
        if cov is None:
            cov = self.cov_mat
        symp = block_diag(self.symplectic_matrix(self.fourier_cutoff), 
                          self.symplectic_matrix(self.fourier_cutoff))
        
        QV = cov + symp*1j/2
        return np.real(eigvals(QV))
            
    def calc_symp_eigvals(self, cov = None):
        if cov is None:
            cov = self.cov_mat
        """Calculating symplectic eigenvalues by exact diagonalization (ED)"""
        if np.shape(cov) == (2*self.fourier_cutoff, 2*self.fourier_cutoff):
            symp = self.symplectic_matrix(self.fourier_cutoff)
            scov = np.matmul(1j*symp, cov)
            
        elif np.shape(cov) == (4*self.fourier_cutoff, 4*self.fourier_cutoff):
            symp = block_diag(self.symplectic_matrix(self.fourier_cutoff), 
                              self.symplectic_matrix(self.fourier_cutoff))
            scov = np.matmul(1j*symp, cov)
        
        #Exact diagonalization
        symp_eigs = eigvals(scov)
        symp_eigs = [np.abs(eig) for eig in symp_eigs if np.real(eig) > 0]
        return symp_eigs
            
    def calc_vN_entropy(self, cov = None, tol = 1e-6):
        """Calculating von Neumann entropy from symplectic eigenvalues"""
        if cov is None:
            cov = self.cov_mat
        symp_eigvals = self.calc_symp_eigvals(cov)
        vN_entropy = 0
        uc = []
        for i in range(len(symp_eigvals)):
            u = symp_eigvals[i]
            if abs(u - 0.5) > 1e-6:
                uc.append(u)
                vN_entropy += (u+0.5)*np.log2(u+0.5) - (u-0.5)*np.log2(u-0.5)
        return vN_entropy
        
    
    def calc_mutual_information(self, cov = None):
        """Calculating (quantum) mutual information from von Neumann entropy"""
        if cov is None:
            cov = self.cov_mat
        reduced_cov_1 = cov[0:2*self.fourier_cutoff, 0:2*self.fourier_cutoff]
        reduced_cov_2 = cov[2*self.fourier_cutoff:4*self.fourier_cutoff, 
                                     2*self.fourier_cutoff:4*self.fourier_cutoff]
        mi = self.calc_vN_entropy(reduced_cov_1)+self.calc_vN_entropy(reduced_cov_2)-self.calc_vN_entropy(cov)
        return mi
    
    
    def partial_transpose(self, cov = None):
        """Partial transposition for PPT Criterion"""
        if cov is None:
            cov = self.cov_mat
        Tb = self.partial_transpose_matrix(self.fourier_cutoff)
        pt_cov = np.matmul(Tb, np.matmul(cov, Tb))
        return pt_cov
    
    def calc_log_negativity(self, cov = None):
        """Compute logarithmic negativity with exact diagonalization"""
        if cov is None:
            cov = self.cov_mat
        pt_cov = self.partial_transpose(cov)
        symp_eigs = self.calc_symp_eigvals(pt_cov)
        log_neg = 0
        for i in range(len(symp_eigs)):
            val = -np.log2(2*symp_eigs[i])
            if val>0:
                log_neg += val
        return log_neg
    
    
    def calc_cond_entropy(self, cov = None):
        """Compute conditional entropy with exact diagonalization"""
        if cov is None:
            cov = self.cov_mat
        
        reduced_cov  = cov[0:2*self.fourier_cutoff, 0:2*self.fourier_cutoff]
        SAB = self.calc_vN_entropy(cov)
        SA = self.calc_vN_entropy(reduced_cov)
        return SAB - SA
    
    
    @staticmethod
    def symplectic_matrix(N):
        """Generate a 2N x 2N standard symplectic matrix with N is the cutoff"""
        I_N = np.eye(N)
        Omega = np.block([[np.zeros((N, N)), -I_N], [I_N, np.zeros((N, N))]])
        return Omega
    
    @staticmethod
    def partial_transpose_matrix(N):
        """Generate 4N x 4N partial transposition matrix"""
        I_N = np.eye(N,N)
        Tb = block_diag(I_N, I_N, -I_N, I_N)
        return Tb