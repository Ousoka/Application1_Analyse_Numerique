# IMPORTATION DES LIBRAIRIES
# ==========================
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.linalg import norm

#Definition du domaine de Calcul
a = 0; b = 1
# Vitesse d'advection
alpha = 10

# MAILLAGE DU DOMAINE

# Nombre de points; points de discrétisation; pas d'espace
NX = 100;x = np.linspace(0.0,1,NX);dx = (b-a)/(NX-1) 

# Choix du Pas de Temps (alpha*dt)/dx = CFL <=1
CFL = 0.5; dt = CFL*dx/alpha; 

# Parametres numerique 
Lambda = (alpha*dt)/dx
ti=0; tf=1
# NT = int(np.floor(1/dt))+1 ; t = 0
NT = int(np.floor((tf-ti)/dt))+1 ; t = 0

#NT = 200


# Condition Initiale
def Func1(x):
    return (x)

U0 = np.vectorize(Func1)

# Condition Limite
def Ulim(t):
    return(np.sin(10*np.pi*t))

# Solution Exacte
def Uex(a,b,alpha,x,t,U0,Ulim):
    return (np.where((x-alpha*t)>=a,U0(x-alpha*t),Ulim(t- (x-a)/alpha)))


#%% Calcul de la Solution Discrete et Test de Stabilité
def U_discret1(x, dx, dt, Alpha, NX, NT, U0):
    U = np.zeros((NX, NT))
    U[:, 0] = U0(x)
    U[0, :] = 0
    U[-1, :] = 0
    N = NX - 1
    gamma = (Alpha * dt) / (dx**2)
    if gamma > 0.5:
        print("Attention: le schéma explicite peut être instable (gamma > 0.5)")
    A = np.diagflat((1 - 2 * gamma) * np.ones(N - 1), k=0) \
        + np.diagflat(gamma * np.ones(N - 2), k=1) + np.diagflat(gamma * np.ones(N - 2), k=-1)
    for n in range(0, NT - 1):
        U[1:N, n + 1] = np.dot(A, U[1:N, n])
    return U

def U_discret2(x, dx, dt, Alpha, NX, NT, U0):
    U = np.zeros((NX, NT))
    U[:, 0] = U0(x)
    U[0, :] = 0
    U[-1, :] = 0
    N = NX - 1
    gamma = (Alpha * dt) / (dx**2)
    A = np.diagflat((1 + 2 * gamma) * np.ones(N - 1), k=0) \
        + np.diagflat(-gamma * np.ones(N - 2), k=1) + np.diagflat(-gamma * np.ones(N - 2), k=-1)
    Ainv = np.linalg.inv(A)
    for n in range(0, NT - 1):
        U[1:N, n + 1] = np.dot(Ainv, U[1:N, n])
    return U



# Données initiales et paramétrage
def u0(x):
    return np.sin(2 * np.pi * x)

# Exemple d'appel des fonctions avec test de stabilité
NX, NT = 10, 10
dx, dt = 1/(NX-1), 0.0001
Alpha = 0.5
x = np.linspace(0, 1, NX)
U_exp = U_discret1(x, dx, dt, Alpha, NX, NT, u0)
U_imp = U_discret2(x, dx, dt, Alpha, NX, NT, u0)


#%% AFFICHAGE DES SOLUTIONS DISCRETES ET EXACTES COMPAREES  

#UU = U_discret1(x,dt,Lambda,NX,NT,U0,Ulim);

UU = U_discret2(x,dx, dt,Alpha,NX,NT,U0)

#UU = U_discret3(x,dt,Lambda,NX,NT,U0,Ulim);

# UU = U_discret4(x,dt,Lambda,NX,NT,U0,Ulim);

t  = 0

for n in range(0,NT):

    Uexact = Uex(a,b,alpha,x,t,U0,Ulim)

    if (n%10 == 0): 
        fig, ax = plt.subplots(1) 
        labe = "Solution discrete a t = %1.2f" %(n * dt); 
        ax.plot(x, U_exp[:,n],'b-o',lw=2, label = labe)
        ax.plot(x, Uexact,'r-+',lw=2, label = 'Solution exacte')
        plt.xlabel('$x$', fontsize=8);plt.ylabel('$U(x,t)$', fontsize=10)
        plt.title('Equation d advection 1D')
        # plt.ylim([-1.1, 1.1])
        # plt.xlim([-0.1, 1.1])
        plt.grid()
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)
        filename = "ADVECT22/Figure_" + str(int(n/10)) + ".pdf"
        #plt.savefig(filename)

    t = t+dt 