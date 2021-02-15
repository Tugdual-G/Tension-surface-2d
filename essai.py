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


# Création du liquide:
P[5:15,8:12] = 1
P[8:13,12] = 1
P[9:12,13] = 1
P[10,14] = 1
P[8:12,8] = 0
P[9:11,9] = 0

# Calcul de la tension:
""" Il vaudrait mieux prendre en compte la courbure 
pour avoir une force nulle sur les lignes droites.
Ici on ne calcule en fait que l'orientation de la surface,
il faudrait calculer et moyenner la variation de cette direction.
"""

T_i[1:-1,:] = P[:-2,:]-P[2:,:]
T_j[:,1:-1] = P[:,:-2]-P[:,2:]


"""On peut essayer de prendre en compte les éléments 
qui se trouvent sur la diagonnale: """
T_i[1:-1,1:-1] += P[:-2,:-2]-P[2:,2:]
T_j[1:-1,1:-1] += P[:-2,:-2]-P[2:,2:]
T_j[1:-1,1:-1] += P[2:,:-2]-P[:-2,2:]
T_i[1:-1,1:-1] -= P[2:,:-2]-P[:-2,2:]

# On suprime la tension pour les points n'appartenants pas au liquide:
T_i = T_i*P
T_j = T_j*P

# Normalisation:
norm = np.sqrt(T_i**2 + T_j**2)
# On peut mettre n'importe quel nombre,1 par exemple,
# afin d'éviter l'erreur division par 0:
norm[norm == 0] = 1

T_i = T_i/norm
T_j = T_j/norm


# Affichage:
plt.figure('test', figsize = (5,5))
plt.clf()
plt.pcolormesh(X, Y, P, shading = 'nearest')
plt.quiver(X, Y, T_j,T_i, scale = 15 )
plt.show()
