import pandas as pd
import matplotlib.pyplot as plt
from itertools import permutations
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

print("--- V1 : RÉSOLUTION PAR DÉNOMBREMENT (BRUTE FORCE) ---")

# ---------------------------------------------------------
# 2. ALGORITHME DE DÉNOMBREMENT
# ---------------------------------------------------------
start_time = time.time()

meilleur_score = -1
meilleure_compo = None
nb_solutions_explorees = 0

# Test de toutes les permutations de 5 joueurs parmi 10
# Complexité : A(10, 5) = 30 240 combinaisons
for equipe in permutations(joueurs, len(postes)):
    nb_solutions_explorees += 1
    score_actuel = 0

    # Calcul du score pour cette permutation
    for i, joueur in enumerate(equipe):
        score_actuel += df.loc[postes[i], joueur]

    if score_actuel > meilleur_score:
        meilleur_score = score_actuel
        meilleure_compo = equipe

end_time = time.time()
execution_time = end_time - start_time

# ---------------------------------------------------------
# 3. RÉSULTATS ET VISUALISATION
# ---------------------------------------------------------
print(f"Solutions explorées : {nb_solutions_explorees}")
print(f"Temps d'exécution   : {execution_time:.4f} sec")
print(f"Score Max trouvé    : {meilleur_score}")

# Préparation graphique
res_postes = postes
res_notes = [df.loc[p, j] for p, j in zip(postes, meilleure_compo)]
res_joueurs = list(meilleure_compo)

plt.figure(figsize=(10, 6))
bars = plt.bar(res_postes, res_notes, color='#3498db')
plt.title(f"V1: Solution Optimale (Dénombrement)\nScore: {meilleur_score} | Temps: {execution_time:.4f}s")
plt.ylabel("Niveau de compétence")
plt.ylim(0, 6)

for bar, nom, note in zip(bars, res_joueurs, res_notes):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.5,
             f"{nom}\n({note})", ha='center', color='white', fontweight='bold')

plt.show()