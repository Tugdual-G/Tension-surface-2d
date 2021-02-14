# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# Nombre de points des grilles:
qi = 20
qj = 20

# Grille d'espace:
x = np.linspace(0,10,qj)
y = np.linspace(0,10,qi)
X, Y = np.meshgrid(x,y)

# P indique la présence du liquide:
P = np.zeros((qi,qj))

# projections de la tension sur x (j) et y (i):
T_i = np.zeros((qi,qj)) 
T_j = np.zeros((qi,qj))

# facteur de force arbitraire:
f = 2

# Création du liquide:
P[5:15,8:12] = 1
P[8:13,12] = 1
P[9:12,13] = 1
P[10,14] = 1
P[8:10,8] = 0


# Calcul de la tension:
T_i[1:-1,:] = f*(P[2:,:] - P[:-2,:])
T_j[:,1:-1] = f*(P[:,2:] - P[:,:-2])
# On suprime la tension pour les points n'appartenants pas au liquide:
T_i = T_i*P
T_j = T_j*P


# Affichage:
plt.figure('test', dpi = 100)
plt.clf()
plt.pcolormesh(X, Y, P, shading = 'nearest')
plt.quiver(X, Y, T_j,T_i, scale = 80 )
plt.show()