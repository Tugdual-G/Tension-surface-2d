# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# On peut travailler à partir d'une image:
"""
img = mpimg.imread('image.png')
img = np.flipud(img[:,:,0])
"""

# Nombre de points des grilles:
qi = 20
qj = qi

# Grille d'espace:
x = np.linspace(0,10,qj)
y = np.linspace(0,10,qi)
X, Y = np.meshgrid(x,y)

# Pas de la grille
h = 10/qi

# P indique la présence du liquide:
P = np.zeros((qi,qj))

# projections du vecteur normal sur x (j) et y (i):
n_i = np.zeros((qi,qj)) 
n_j = np.zeros((qi,qj))


# Création du liquide:

P[5:15,5:12] = 1
P[8:13,12] = 1
P[9:12,13] = 1
P[11,14] = 1
P[9:12,5:6] = 0
"""
P = img
"""
# =============================================================================
#                          Calcul du vecteur normal
# =============================================================================
n_i[1:-1,:] = P[:-2,:]-P[2:,:]
n_j[:,1:-1] = P[:,:-2]-P[:,2:]

# On supprime la normale pour les points n'appartenant pas au liquide:
n_i = n_i*P
n_j = n_j*P

# Extraction de la norme pour normalisation:
norm = np.sqrt(n_i**2 + n_j**2)
norm[norm == 0] = 1e-10

# S indique les points du contour, vaut 1 à leur emplacement, ailleurs: 
S = np.where(norm > 1e-10, 1,0)

# Normalisation:
n_i = S*n_i/norm
n_j = S*n_j/norm

# =============================================================================
#                   Fonctions de base du calcul
# =============================================================================

def moyenne_n(n,S):
    """Calcule la moyenne pondérée de la normale en chaque point, à partir des
    points adjacents."""
    # Nombre de points du contour dans le voisinage:    
    nbr_prox = (S[2:,2:] + S[2:,1:-1] + S[2:,:-2] + S[:-2,:-2] +
                S[:-2, 1:-1] + S[:-2,2:] + S[1:-1,:-2] + S[1:-1, 2:])
    # Afin d'éviter division par 0:
    nbr_prox[nbr_prox == 0] = 1
    
    # Moyenne pondérée:
    n[1:-1, 1:-1] = S[1:-1, 1:-1]*(n[2:,2:] + n[2:,1:-1] + n[2:,:-2] +
                       n[:-2,:-2] + n[:-2, 1:-1] + n[:-2,2:] +
                       n[1:-1,:-2] + n[1:-1, 2:])*0.5/nbr_prox + n[1:-1, 1:-1]*0.5
    return n


def grad_courbe(ni, nj, S, h, P): 
    """Calcule la variation de direction de la normale le long de la surface
    indiqué par S. On considère que la somme de ces variations est une façon
    de quantifier la courbure, courbure qui peut être relié à la force par un
    facteur, ce facteur est 1 pour le moment."""
    
    # k correspond au vecteur unitaire tangent à la surface:
    k_j = -n_i
    k_i = n_j
    
    # Matrice donnant le sens de parcours pour la dérivation: sens_i, sens_j
    norm = np.abs(k_i)
    norm[norm == 0] = 1
    sens_i = k_i/norm
    
    norm = np.abs(k_j)
    norm[norm == 0] = 1
    sens_j = k_j/norm
    
    # Dérivation selon haut et bas (i), gauche et droite(j):
    i_dk_nk = (k_i[1:-1,1:-1]*(n_i[2:,1:-1]-n_i[:-2,1:-1])+
              k_j[1:-1,1:-1]*(n_j[2:,1:-1]-n_j[:-2,1:-1]))*sens_i[1:-1,1:-1]
    
    j_dk_nk = (k_i[1:-1,1:-1]*(n_i[1:-1,2:]-n_i[1:-1,:-2])+
              k_j[1:-1,1:-1]*(n_j[1:-1,2:]-n_j[1:-1,:-2]))*sens_j[1:-1,1:-1] 
     
    #********* Dérivation en diagonale
    # Sens de dérivation selon les deux diagonales:
    sens_dg1 = k_i + k_j
    norm = np.abs(sens_dg1)
    norm[norm == 0] = 1
    sens_dg1 = sens_dg1/norm 
    
    sens_dg2 = k_j - k_i
    norm = np.abs(sens_dg2)
    norm[norm == 0] = 1
    sens_dg2 = sens_dg2/norm     
    
    # Diag indique s'il faut dériver ou non selon la diagonale.
    diag = (1-S[2:,1:-1]*S[:-2,1:-1])*(1-S[1:-1,2:]*S[1:-1,:-2])
    
    # Dérivation:
    dg1_dk_nk = (k_i[1:-1,1:-1]*(n_i[2:,2:]-n_i[:-2,:-2])+
              k_j[1:-1,1:-1]*(n_j[2:,2:]-n_j[:-2,:-2]))*sens_dg1[1:-1,1:-1]*diag 
    
    dg2_dk_nk = (k_i[1:-1,1:-1]*(n_i[:-2,2:]-n_i[2:,:-2])+
          k_j[1:-1,1:-1]*(n_j[:-2,2:]-n_j[2:,:-2]))*sens_dg2[1:-1,1:-1]*diag
    
    #*********** Calcul de la force
    force = n_i*0
    force[1:-1,1:-1] = (i_dk_nk + j_dk_nk + dg1_dk_nk + dg2_dk_nk)/(2*h)          
    return force

# =============================================================================
#                                   Calcul
# =============================================================================
    
n_i = moyenne_n(n_i, S)
n_j = moyenne_n(n_j, S)
force = grad_courbe(n_i, n_j, S, h, P) 

# =============================================================================
#                                 Affichage
# =============================================================================
plt.figure('test', figsize = (10,10))
plt.clf()
plt.pcolormesh(X, Y, P+ force, shading = 'nearest')
plt.quiver(X, Y, -force*n_j, -force*n_i, scale = 25)
plt.show()
