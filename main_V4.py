import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import time

# ---------------------------------------------------------
# 1. RÉCUPÉRATION DES DONNÉES (Mêmes paramètres que V3)
# ---------------------------------------------------------
np.random.seed(42)
NB_POSTES = 50
NB_CANDIDATS = 100

# Génération (Notes 1-20)
data = np.random.randint(1, 21, size=(NB_POSTES, NB_CANDIDATS))
df = pd.DataFrame(data, 
                  index=[f"P{i+1}" for i in range(NB_POSTES)], 
                  columns=[f"C{i+1}" for i in range(NB_CANDIDATS)])

print(f"--- LANCEMENT V4 : OPTIMISATION LINÉAIRE ---")
print(f"Taille du problème : {NB_POSTES} Postes x {NB_CANDIDATS} Candidats")
print(f"Nombre de variables binaires : {NB_POSTES * NB_CANDIDATS}")

# ---------------------------------------------------------
# 2. MODÉLISATION MATHÉMATIQUE
# ---------------------------------------------------------
start_time = time.time()

# A. Fonction Objectif (Maximiser les scores -> Minimiser les négatifs)
c = -1 * df.values.flatten()

# B. Contraintes d'Égalité : Chaque poste doit avoir EXACTEMENT 1 personne
# Matrice A_eq de taille (50 lignes, 5000 colonnes)
A_eq = np.zeros((NB_POSTES, NB_POSTES * NB_CANDIDATS))
b_eq = np.ones(NB_POSTES)

for i in range(NB_POSTES):
    # Les colonnes correspondant au poste i sont mises à 1
    A_eq[i, i*NB_CANDIDATS : (i+1)*NB_CANDIDATS] = 1

# C. Contraintes d'Inégalité : Chaque candidat ne peut prendre qu'au MAX 1 poste
# Matrice A_ub de taille (100 lignes, 5000 colonnes)
A_ub = np.zeros((NB_CANDIDATS, NB_POSTES * NB_CANDIDATS))
b_ub = np.ones(NB_CANDIDATS)

for j in range(NB_CANDIDATS): # Pour chaque candidat
    for i in range(NB_POSTES): # Sur tous les postes possibles
        # On marque la colonne correspondant au (Poste i, Candidat j)
        A_ub[j, i*NB_CANDIDATS + j] = 1

# ---------------------------------------------------------
# 3. RÉSOLUTION (SOLVEUR)
# ---------------------------------------------------------
print("Calcul en cours avec l'algorithme 'Highs'...")

res = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=(0, 1), method='highs')

end_time = time.time()

# ---------------------------------------------------------
# 4. RÉSULTATS ET VISUALISATION (SANS SEABORN)
# ---------------------------------------------------------
if res.success:
    score_total = -res.fun
    print(f"\n🚀 OPTIMISATION RÉUSSIE en {end_time - start_time:.4f} secondes !")
    print(f"Score Total Maximisé : {score_total:.0f} / {NB_POSTES * 20} (max théorique)")
    
    choix_matrix = res.x.reshape(NB_POSTES, NB_CANDIDATS).round().astype(int)

    plt.figure(figsize=(14, 8))
    
    # Heatmap simple matplotlib
    plt.imshow(df.values, aspect='auto')
    plt.colorbar(label="Note (1-20)")

    # Points rouges pour les affectations
    rows, cols = np.where(choix_matrix == 1)
    plt.scatter(cols, rows, s=30)

    plt.title(f"Optimisation V4 : Affectation de {NB_POSTES} postes parmi {NB_CANDIDATS} candidats\nScore Total : {score_total:.0f}", fontsize=16)
    plt.xlabel(f"Les {NB_CANDIDATS} Candidats")
    plt.ylabel(f"Les {NB_POSTES} Postes")

    plt.tight_layout()
    plt.show()
