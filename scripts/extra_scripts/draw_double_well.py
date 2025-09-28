#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 30 18:31:36 2025

@author: taufiqmurtadho
"""

import numpy as np
import matplotlib.pyplot as plt

# Define spatial coordinate
x = np.linspace(-2.5, 2.5, 500)

# Define the double-well potential with higher barrier
V = 50 * (x**2 - 1)**2  # High barrier

# Define a fixed probability distribution (same as before)
psi = 5*np.exp(-10*(x + 1)**2) + 5*np.exp(-10*(x - 1)**2)
#psi /= np.trapz(psi, x)  # Normalize
psi *= 10  # Scale for better visibility in plot

# Plotting
fig, ax = plt.subplots(figsize=(4, 6))

# Plot probability distribution
ax.fill_betweenx(x, 0, psi, color='indianred', alpha = 0.7)

# Plot potential (vertical profile)
ax.plot(V, x, 'k', linewidth=3)

# Adjust limits so the full potential plateau is visible
ax.set_xlim(-0.5, np.max(V)*0.075)  # Adjust this scaling factor as needed
ax.axis('off')
plt.tight_layout()

#plt.savefig('../figures/double_well.pdf', format='pdf', dpi=1200)
