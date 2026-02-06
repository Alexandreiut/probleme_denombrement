import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
import time

# ---------------------------------------------------------
# 1. CRÉATION DES DONNÉES (D'après votre image Excel)
# ---------------------------------------------------------

# Les joueurs (A à J) et leurs notes par poste
data = {
    'A': [4, 5, 3, 1, 3],
    'B': [1, 1, 1, 2, 3],
    'C': [3, 1, 2, 2, 1],
    'D': [3, 4, 1, 1, 2],
    'E': [1, 2, 2, 3, 4],
    'F': [5, 3, 3, 4, 2],
    'G': [4, 2, 2, 2, 3],
    'H': [2, 2, 1, 2, 1],
    'I': [1, 1, 2, 2, 4],
    'J': [2, 2, 1, 1, 2]
}

postes = ['Avant', 'Milieu', 'Ailier gauche', 'Ailier droit', 'Arrière']
joueurs = list(data.keys()) # ['A', 'B', 'C', ...]

# Création du DataFrame Pandas pour une belle présentation
df = pd.DataFrame(data, index=postes)

print("--- Tableau des scores (Matrice des compétences) ---")
print(df)
print("-" * 50)

# ---------------------------------------------------------
# 2. RÉSOLUTION PAR DÉNOMBREMENT (BRUTE FORCE)
# ---------------------------------------------------------

# En mathématiques, on cherche un arrangement de 5 joueurs parmi 10.
# Formule : A(10, 5) = 10 * 9 * 8 * 7 * 6 = 30 240 possibilités.

start_time = time.time()

meilleur_score = 0
meilleure_equipe = None
compteur = 0

# On génère toutes les permutations possibles de 5 joueurs parmi les 10 disponibles
# Chaque permutation est un tuple, ex: ('C', 'A', 'F', 'E', 'I')
# La position 0 du tuple correspond au poste 'Avant', la 1 à 'Milieu', etc.
for equipe_test in permutations(joueurs, 5):
    compteur += 1
    score_actuel = 0
    
    # Calcul du score de cette équipe
    for i, joueur in enumerate(equipe_test):
        poste = postes[i]
        note = df.loc[poste, joueur]
        score_actuel += note
    
    # Vérification si c'est le record
    if score_actuel > meilleur_score:
        meilleur_score = score_actuel
        meilleure_equipe = equipe_test

end_time = time.time()

# ---------------------------------------------------------
# 3. AFFICHAGE DES RÉSULTATS
# ---------------------------------------------------------

print(f"Nombre total d'équipes testées : {compteur}")
print(f"Temps de calcul : {end_time - start_time:.4f} secondes")
print(f"\n SCORE MAXIMAL TROUVÉ (Zmax) : {meilleur_score}")
print("\nComposition de l'équipe optimale :")

equipe_optimale_dict = {}
for i, poste in enumerate(postes):
    joueur = meilleure_equipe[i]
    note = df.loc[poste, joueur]
    equipe_optimale_dict[poste] = {'Joueur': joueur, 'Note': note}
    print(f"- {poste:15s} : Joueur {joueur} (Note : {note})")

# ---------------------------------------------------------
# 4. VISUALISATION GRAPHIQUE
# ---------------------------------------------------------

# Préparation des données pour le graph
postes_labels = list(equipe_optimale_dict.keys())
notes_valeurs = [d['Note'] for d in equipe_optimale_dict.values()]
noms_joueurs = [d['Joueur'] for d in equipe_optimale_dict.values()]

# Création des couleurs : vert si note excellente (5), bleu sinon
couleurs = ['#2ecc71' if n == 5 else '#3498db' for n in notes_valeurs]

plt.figure(figsize=(10, 6))
barres = plt.bar(postes_labels, notes_valeurs, color=couleurs)

# Ajout des noms des joueurs sur les barres
for i, barre in enumerate(barres):
    plt.text(barre.get_x() + barre.get_width()/2, 
             barre.get_height() - 0.5, 
             f"J.{noms_joueurs[i]}\n({notes_valeurs[i]})", 
             ha='center', va='center', color='white', fontweight='bold', fontsize=12)

plt.title(f"Composition de l'équipe Optimale (Score Total: {meilleur_score})", fontsize=16)
plt.ylabel('Note du joueur (1-5)', fontsize=12)
plt.ylim(0, 6)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Sauvegarder ou afficher
# plt.savefig('resultat_equipe.png') 
plt.show()