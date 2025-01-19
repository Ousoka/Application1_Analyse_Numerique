# IMPORTATION DES LIBRAIRIES
# ==========================
import numpy as np
from sympy import symbols, Function, Eq, dsolve, sinh
from sympy.abc import x
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


# ETAPE 1 : SOLUTION ANALYTIQUE
# =============================
# On considère l'équation différentielle stationnaire :
# -mu * u''(x) + gamma * u(x) = 0, avec les conditions aux limites u(0) = 0 et u(1) = 1.

def solution_analytique(x, mu, gamma):
    """
    Fonction pour calculer la solution analytique u(x).
    """
    lambda_ = np.sqrt(gamma / mu)
    return np.sinh(lambda_ * x) / np.sinh(lambda_)


# ETAPE 2 : MAILLAGE DU DOMAINE
# =============================
# Discrétisation du domaine spatial avec un pas dx.

def maillage_domaine(NX):
    """
    Génère le maillage du domaine spatial.
    """
    x = np.linspace(0, 1, NX)  # Positions discrètes sur le domaine [0,1]
    dx = 1 / (NX - 1)  # Pas de discrétisation
    return x, dx


# ETAPE 3 : FONCTION POUR LA SOLUTION NUMERIQUE
# =============================================
# Utilisation de la méthode des différences finies pour résoudre l'EDO.

def solution_numerique(NX, mu, gamma):
    """
    Calcule la solution numérique de l'équation par différences finies.
    """
    # Maillage du domaine
    x, dx = maillage_domaine(NX)

    # Construction de la matrice tridiagonale
    alpha = mu / dx**2
    beta = gamma
    main_diag = 2 * alpha + beta  # Diagonale principale
    lower_upper_diag = -alpha  # Diagonales inférieure et supérieure

    diagonals = [lower_upper_diag * np.ones(NX - 2),  # Diagonale inférieure
                 main_diag * np.ones(NX - 2),  # Diagonale principale
                 lower_upper_diag * np.ones(NX - 2)]  # Diagonale supérieure

    A = diags(diagonals, offsets=[-1, 0, 1], format='csr')

    # Construction du vecteur source F
    F = np.zeros(NX - 2)  # Initialisation du vecteur source
    F[-1] = alpha  # Contribution de la condition aux limites u(1) = 1

    # Résolution du système linéaire
    U_internal = spsolve(A, F)

    # Ajout des conditions aux limites
    U = np.zeros(NX)
    U[1:NX-1] = U_internal
    U[-1] = 1  # Condition limite u(1) = 1

    return x, U


# ETAPE 4 : AFFICHAGE ET COMPARAISON DES SOLUTIONS
# ===============================================
# Tracer la solution analytique et la solution numérique.

def affichage_comparaison(NX, mu, gamma):
    """
    Affiche la comparaison entre la solution analytique et numérique.
    """
    # Calcul des solutions
    x, dx = maillage_domaine(NX)
    x_num, U_num = solution_numerique(NX, mu, gamma)
    U_analytique = solution_analytique(x, mu, gamma)

    # Tracé des courbes
    plt.figure(figsize=(8, 6))
    plt.plot(x, U_analytique, 'b-', label="Solution analytique", linewidth=2)
    plt.plot(x_num, U_num, 'r--+', label="Solution numérique", linewidth=2)
    plt.title("Comparaison des solutions (analytique vs numérique)", fontsize=14)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("u(x)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()


# ETAPE 5 : TESTS ET ANALYSES DES CAS
# ===================================
# Étude des cas :
# (a) gamma / (6 * mu) << 1  : Diffusion domine.
# (b) gamma / (6 * mu) >> 1  : Réaction domine.

def analyse_cas():
    """
    Analyse les différents cas de rapport gamma / mu.
    """
    print("Cas (a) : Diffusion domine (gamma / (6 * mu) << 1)")
    affichage_comparaison(NX=50, mu=1, gamma=0.1)

    print("Cas (b) : Réaction domine (gamma / (6 * mu) >> 1)")
    affichage_comparaison(NX=50, mu=1, gamma=100)


# ETAPE 6 : STABILITE NUMERIQUE
# =============================
# Le critère de stabilité dépend du rapport h^2 * gamma / mu. Si ce rapport est trop grand,
# la solution devient instable.

def analyse_stabilite():
    """
    Vérifie le critère de stabilité pour différents pas dx.
    """
    mu = 1
    gamma = 100
    for NX in [10, 20, 50, 100, 200]:
        x, dx = maillage_domaine(NX)
        print(f"Pour NX = {NX}, dx = {dx:.5f}, Rapport (h^2 * gamma / mu) = {dx**2 * gamma / mu:.5f}")
        affichage_comparaison(NX, mu, gamma)


# MAIN : EXECUTION DES DIFFERENTES ANALYSES
# =========================================
if __name__ == "__main__":
    print("== COMPARAISON SOLUTIONS ==")
    affichage_comparaison(NX=50, mu=1, gamma=10)

    print("\n== ANALYSE DES CAS ==")
    analyse_cas()

    print("\n== STABILITE NUMERIQUE ==")
    analyse_stabilite()
