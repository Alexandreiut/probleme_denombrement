import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
import time

# ---------------------------------------------------------
# 1. GÉNÉRATION DE DONNÉES ALÉATOIRES
# ---------------------------------------------------------
np.random.seed(42)
# PARAMÈTRES INPUT
NB_POSTES = 7
NB_JOUEURS = 14  # Ratio 2:1

data = np.random.randint(1, 21, size=(NB_POSTES, NB_JOUEURS))
df = pd.DataFrame(data,
                  index=[f"P{i + 1}" for i in range(NB_POSTES)],
                  columns=[f"J{i + 1}" for i in range(NB_JOUEURS)])

print(f"--- V3 : DÉNOMBREMENT SCALABLE ({NB_POSTES} postes x {NB_JOUEURS} joueurs) ---")
print("Note : Sur de grandes tailles, cet algorithme est extrêmement lent.")

# ---------------------------------------------------------
# 2. ALGORITHME
# ---------------------------------------------------------
start_time = time.time()

meilleur_score = -1
meilleure_compo_indices = None
nb_solutions = 0
indices_joueurs = range(NB_JOUEURS)

# Permutations des indices des joueurs (pour aller plus vite que les strings)
for equipe_indices in permutations(indices_joueurs, NB_POSTES):
    nb_solutions += 1
    score_actuel = 0

    for i, idx_j in enumerate(equipe_indices):
        score_actuel += data[i, idx_j]

    if score_actuel > meilleur_score:
        meilleur_score = score_actuel
        meilleure_compo_indices = equipe_indices

end_time = time.time()
execution_time = end_time - start_time

# ---------------------------------------------------------
# 3. VISUALISATION
# ---------------------------------------------------------
print(f"Combinaisons testées : {nb_solutions}")
print(f"Temps d'exécution    : {execution_time:.4f} sec")
print(f"Score Max            : {meilleur_score}")

plt.figure(figsize=(10, 6))
plt.imshow(df.values, cmap='Blues', aspect='auto')
plt.colorbar(label="Note (1-20)")

# Marquage des solutions
for i, idx_j in enumerate(meilleure_compo_indices):
    plt.scatter(idx_j, i, color='red', s=100, label='Sélection' if i == 0 else "")

plt.title(f"V3: Matrice de décision (Brute Force)\nExploré: {nb_solutions} solutions en {execution_time:.4f}s")
plt.xlabel("Joueurs")
plt.ylabel("Postes")
plt.legend(loc='upper right')
plt.show()