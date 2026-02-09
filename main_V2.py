import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# ---------------------------------------------------------
# 1. DONNÉES (Matrice des scores C)
# ---------------------------------------------------------
# Reproduit le tableau exact de votre image
data = {
    'A': [4, 5, 3, 1, 3],
    'B': [1, 1, 1, 2, 3],
    'C': [5, 1, 2, 2, 1],
    'D': [3, 4, 1, 1, 2],
    'E': [1, 2, 2, 3, 4],
    'F': [5, 3, 3, 4, 2],
    'G': [4, 2, 2, 2, 3],
    'H': [2, 2, 1, 2, 1],
    'I': [1, 1, 2, 2, 4],
    'J': [2, 2, 1, 1, 2]
}
postes = ['Avant', 'Milieu', 'Ailier G', 'Ailier D', 'Arrière']
joueurs = list(data.keys())

# DataFrame pour affichage (Matrice des Coûts/Gains)
df_scores = pd.DataFrame(data, index=postes)

print("--- Matrice des scores ---")
print(df_scores)
print("-" * 50)

# ---------------------------------------------------------
# 2. MODÉLISATION POUR LE SOLVEUR (SCIPY)
# ---------------------------------------------------------

# A. Fonction Objectif
# Le solveur cherche à MINIMISER. Pour MAXIMISER le score, 
# on passe les valeurs en négatif.
# On "aplatit" la matrice (5 postes * 10 joueurs = 50 variables)
couts = -1 * df_scores.values.flatten() 

# B. Contraintes d'Égalité (A_eq) : Chaque poste doit avoir EXACTEMENT 1 joueur
# On a 5 postes. Pour chaque poste, la somme des variables des 10 joueurs = 1.
A_eq = np.zeros((5, 50))
b_eq = np.ones(5)

for i in range(5): # Pour chaque poste
    # On met des 1 pour les 10 joueurs correspondant à ce poste
    A_eq[i, i*10 : (i+1)*10] = 1

# C. Contraintes d'Inégalité (A_ub) : Chaque joueur a au MAXIMUM 1 poste
# On a 10 joueurs. La somme de leurs apparitions sur les 5 postes <= 1.
A_ub = np.zeros((10, 50))
b_ub = np.ones(10)

for j in range(10): # Pour chaque joueur
    for i in range(5): # Pour chaque poste
        # On sélectionne la variable correspondant au joueur j au poste i
        A_ub[j, i*10 + j] = 1

# ---------------------------------------------------------
# 3. RÉSOLUTION
# ---------------------------------------------------------
print("Lancement de l'optimisation linéaire (Méthode du Simplexe)...")

resultat = linprog(c=couts,          # Fonction objectif
                   A_eq=A_eq, b_eq=b_eq,  # Contraintes postes (=1)
                   A_ub=A_ub, b_ub=b_ub,  # Contraintes joueurs (<=1)
                   bounds=(0, 1),         # Variables binaires (entre 0 et 1)
                   method='highs')        # Algorithme moderne

# ---------------------------------------------------------
# 4. INTERPRÉTATION DES RÉSULTATS
# ---------------------------------------------------------

if resultat.success:
    # On remet le résultat (vecteur de 50) sous forme de matrice 5x10
    choix_optimal = resultat.x.reshape(5, 10)
    
    # À cause des calculs flottants, on arrondit (0.9999 -> 1)
    choix_optimal = np.round(choix_optimal).astype(int)
    
    score_total = -resultat.fun # On remet le score en positif
    print(f"\n✅ Solution Optimale trouvée ! Score Total : {score_total:.0f}")
    
    # Création du tableau final
    equipe_finale = []
    
    print("\nComposition de l'équipe :")
    for i, poste in enumerate(postes):
        # On cherche l'index du joueur sélectionné (là où il y a un 1)
        idx_joueur = np.argmax(choix_optimal[i])
        nom_joueur = joueurs[idx_joueur]
        note = df_scores.iloc[i, idx_joueur]
        
        equipe_finale.append({'Poste': poste, 'Joueur': nom_joueur, 'Note': note})
        print(f"- {poste:15s} : Joueur {nom_joueur} (Note : {note})")
        
else:
    print("Pas de solution trouvée.")

# ---------------------------------------------------------
# 5. VISUALISATION
# ---------------------------------------------------------
df_res = pd.DataFrame(equipe_finale)

plt.figure(figsize=(10, 5))
couleurs = ['green' if x == 5 else 'skyblue' for x in df_res['Note']]
bars = plt.bar(df_res['Poste'], df_res['Note'], color=couleurs)

plt.title(f"Résultat de l'Optimisation Linéaire (Z = {score_total:.0f})")
plt.ylim(0, 6)
plt.ylabel('Note')

# Ajouter le nom du joueur sur la barre
for bar, joueur, note in zip(bars, df_res['Joueur'], df_res['Note']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.5, 
             f"{joueur}\n({note})", ha='center', color='white', fontweight='bold')

plt.show()
