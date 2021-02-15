# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

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



# =============================================================================
#                          Calcul du vecteur normal
# =============================================================================
n_i[1:-1,:] = P[:-2,:]-P[2:,:]
n_j[:,1:-1] = P[:,:-2]-P[:,2:]


# On suprime la normale pour les points n'appartenants pas au liquide:
n_i = n_i*P
n_j = n_j*P

# Extraction de la norme pour normalisation:
norm = np.sqrt(n_i**2 + n_j**2)
norm[norm == 0] = 1e-10

# S indique les points du contour, vaut 1 à leur emplacement, ailleur: 
S = np.where(norm > 1e-10, 1,0)

# Normalisation:
n_i = S*n_i/norm
n_j = S*n_j/norm


# =============================================================================
#                   Fonctions de bases du calcul
# =============================================================================

def moyenne_n(n,S):    
    nbr_prox = (S[2:,2:] + S[2:,1:-1] + S[2:,:-2] + S[:-2,:-2] +
                S[:-2, 1:-1] + S[:-2,2:] + S[1:-1,:-2] + S[1:-1, 2:])
    nbr_prox[nbr_prox == 0] = 1
    
    n[1:-1, 1:-1] = S[1:-1, 1:-1]*(n[2:,2:] + n[2:,1:-1] + n[2:,:-2] +
                       n[:-2,:-2] + n[:-2, 1:-1] + n[:-2,2:] +
                       n[1:-1,:-2] + n[1:-1, 2:])*0.4/nbr_prox + n[1:-1, 1:-1]*0.6
    return n


def grad_courbe(ni, nj ,S,h, P): 
    
    k_j = -n_i
    k_i = n_j
    
    norm = np.abs(k_i)
    norm[norm == 0] = 1
    sens_i = k_i/norm
    norm = np.abs(k_j)
    norm[norm == 0] = 1
    sens_j = k_j/norm
    
    force = n_i*0
    
    i_dk_nk = (k_i[1:-1,1:-1]*(n_i[2:,1:-1]-n_i[:-2,1:-1])+
              k_j[1:-1,1:-1]*(n_j[2:,1:-1]-n_j[:-2,1:-1]))*sens_i[1:-1,1:-1]
    
    j_dk_nk = (k_i[1:-1,1:-1]*(n_i[1:-1,2:]-n_i[1:-1,:-2])+
              k_j[1:-1,1:-1]*(n_j[1:-1,2:]-n_j[1:-1,:-2]))*sens_j[1:-1,1:-1] 
     
    sens_dg1 = k_i + k_j
    norm = np.abs(sens_dg1)
    norm[norm == 0] = 1
    sens_dg1 = sens_dg1/norm 
    
    sens_dg2 = k_j - k_i
    norm = np.abs(sens_dg2)
    norm[norm == 0] = 1
    sens_dg2 = sens_dg2/norm     
    
    diag = (1-S[2:,1:-1]*S[:-2,1:-1])*(1-S[1:-1,2:]*S[1:-1,:-2])
    
    dg1_dk_nk = (k_i[1:-1,1:-1]*(n_i[2:,2:]-n_i[:-2,:-2])+
              k_j[1:-1,1:-1]*(n_j[2:,2:]-n_j[:-2,:-2]))*sens_dg1[1:-1,1:-1]*diag 
    
    dg2_dk_nk = (k_i[1:-1,1:-1]*(n_i[:-2,2:]-n_i[2:,:-2])+
          k_j[1:-1,1:-1]*(n_j[:-2,2:]-n_j[2:,:-2]))*sens_dg2[1:-1,1:-1]*diag
    
    force[1:-1,1:-1] = (i_dk_nk + j_dk_nk + dg1_dk_nk + dg2_dk_nk)/(2*h)          
    
    return force



# =============================================================================
#                                   Calcul
# =============================================================================
    
n_i = moyenne_n(n_i, S)
n_j = moyenne_n(n_j, S)
force = grad_courbe(n_i, n_j, S, h, P) 


# =============================================================================
#                                 Affichage
# =============================================================================
plt.figure('test0.7', figsize = (5,5))
plt.clf()
plt.pcolormesh(X, Y, P+ force, shading = 'nearest')
plt.quiver(X, Y, -force*n_j, -force*n_i, scale = 15)
plt.show()