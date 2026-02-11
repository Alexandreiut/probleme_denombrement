import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import time

# ---------------------------------------------------------
# 1. GÉNÉRATION DE DONNÉES ALÉATOIRES (GRANDE ÉCHELLE)
# ---------------------------------------------------------
np.random.seed(42)
NB_POSTES = 8
NB_JOUEURS = 12 # Ratio 2:1

data = np.random.randint(1, 21, size=(NB_POSTES, NB_JOUEURS))
df = pd.DataFrame(data,
                  index=[f"P{i+1}" for i in range(NB_POSTES)],
                  columns=[f"J{i+1}" for i in range(NB_JOUEURS)])

print(f"--- V4 : SIMPLEXE SCALABLE ({NB_POSTES} postes x {NB_JOUEURS} joueurs) ---")

# ---------------------------------------------------------
# 2. ALGORITHME (OPTIMISATION)
# ---------------------------------------------------------
start_time = time.time()

# A. Fonction Objectif
c = -1 * df.values.flatten()

# B. Contraintes Égalité (Postes)
A_eq = np.zeros((NB_POSTES, NB_POSTES * NB_JOUEURS))
for i in range(NB_POSTES):
    A_eq[i, i*NB_JOUEURS : (i+1)*NB_JOUEURS] = 1
b_eq = np.ones(NB_POSTES)

# C. Contraintes Inégalité (Joueurs)
A_ub = np.zeros((NB_JOUEURS, NB_POSTES * NB_JOUEURS))
for j in range(NB_JOUEURS):
    for i in range(NB_POSTES):
        A_ub[j, i*NB_JOUEURS + j] = 1
b_ub = np.ones(NB_JOUEURS)

# Résolution
res = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=(0, 1), method='highs')

end_time = time.time()
execution_time = end_time - start_time

# ---------------------------------------------------------
# 3. VISUALISATION
# ---------------------------------------------------------
score_total = -res.fun
choix = res.x.reshape(NB_POSTES, NB_JOUEURS).round().astype(int)

print(f"Itérations solveur : {res.nit}")
print(f"Temps d'exécution  : {execution_time:.4f} sec")
print(f"Score Max          : {score_total:.0f}")

plt.figure(figsize=(10, 6))
plt.imshow(df.values, cmap='Greens', aspect='auto') # Vert pour Simplexe
plt.colorbar(label="Note (1-20)")

# Marquage des points sélectionnés
rows, cols = np.where(choix == 1)
plt.scatter(cols, rows, color='red', s=15, label='Optimisé')

plt.title(f"V4: Matrice de décision (Simplexe) - {NB_POSTES}x{NB_JOUEURS}\nCalculé en {execution_time:.4f}s")
plt.xlabel("Joueurs")
plt.ylabel("Postes")
plt.show()