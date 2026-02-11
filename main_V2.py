import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import time

# ---------------------------------------------------------
# 1. DONNÉES ET CONFIGURATION
# ---------------------------------------------------------
data = {
    'A': [4, 5, 3, 1, 3], 'B': [1, 1, 1, 2, 3],
    'C': [5, 1, 2, 2, 1], 'D': [3, 4, 1, 1, 2],
    'E': [1, 2, 2, 3, 4], 'F': [5, 3, 3, 4, 2],
    'G': [4, 2, 2, 2, 3], 'H': [2, 2, 1, 2, 1],
    'I': [1, 1, 2, 2, 4], 'J': [2, 2, 1, 1, 2]
}
postes = ['Avant', 'Milieu', 'Ailier G', 'Ailier D', 'Arrière']
joueurs = list(data.keys())
df = pd.DataFrame(data, index=postes)
nb_postes, nb_joueurs = df.shape

print("--- V2 : RÉSOLUTION PAR SIMPLEXE (OPTIMISATION LINÉAIRE) ---")

# ---------------------------------------------------------
# 2. MODÉLISATION ET RÉSOLUTION
# ---------------------------------------------------------
start_time = time.time()

# A. Fonction Objectif : Minimiser (-Scores) pour Maximiser (Scores)
c = -1 * df.values.flatten()

# B. Contraintes d'Égalité : 1 joueur par poste exact
A_eq = np.zeros((nb_postes, nb_postes * nb_joueurs))
for i in range(nb_postes):
    A_eq[i, i*nb_joueurs : (i+1)*nb_joueurs] = 1
b_eq = np.ones(nb_postes)

# C. Contraintes d'Inégalité : Max 1 poste par joueur
A_ub = np.zeros((nb_joueurs, nb_postes * nb_joueurs))
for j in range(nb_joueurs):
    for i in range(nb_postes):
        A_ub[j, i*nb_joueurs + j] = 1
b_ub = np.ones(nb_joueurs)

# Résolution
res = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=(0, 1), method='highs')

end_time = time.time()
execution_time = end_time - start_time

# ---------------------------------------------------------
# 3. RÉSULTATS ET VISUALISATION
# ---------------------------------------------------------
# Le nombre d'itérations correspond aux pivots du simplexe (explorations intelligentes)
nb_iterations = res.nit
score_max = -res.fun

print(f"Itérations (Pivots) : {nb_iterations}")
print(f"Temps d'exécution   : {execution_time:.4f} sec")
print(f"Score Max trouvé    : {score_max:.0f}")

# Reconstruction de la solution
choix = res.x.reshape(nb_postes, nb_joueurs).round().astype(int)
res_notes = []
res_joueurs_noms = []

for i in range(nb_postes):
    idx_j = np.argmax(choix[i]) # Trouve l'index du joueur choisi (le 1)
    res_joueurs_noms.append(joueurs[idx_j])
    res_notes.append(df.iloc[i, idx_j])

plt.figure(figsize=(10, 6))
bars = plt.bar(postes, res_notes, color='#2ecc71') # Vert pour différencier du brute force
plt.title(f"V2: Solution Optimale (Simplexe)\nScore: {score_max:.0f} | Temps: {execution_time:.4f}s")
plt.ylabel("Niveau de compétence")
plt.ylim(0, 6)

for bar, nom, note in zip(bars, res_joueurs_noms, res_notes):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.5,
             f"{nom}\n({note})", ha='center', color='white', fontweight='bold')

plt.show()