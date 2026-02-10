import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import permutations
import time

# ---------------------------------------------------------
# 1. GÉNÉRATION DE DONNÉES COMPLEXES (ALÉATOIRE)
# ---------------------------------------------------------
np.random.seed(42) # Pour avoir les mêmes résultats à chaque fois

# Imaginons une "Armée" ou une "Grande Entreprise"
NB_POSTES_TOTAL = 50
NB_CANDIDATS_TOTAL = 100

# Génération des notes entre 1 et 20
data = np.random.randint(1, 21, size=(NB_POSTES_TOTAL, NB_CANDIDATS_TOTAL))

# Création des labels
postes_labels = [f"Poste_{i+1}" for i in range(NB_POSTES_TOTAL)]
candidats_labels = [f"Candidat_{i+1}" for i in range(NB_CANDIDATS_TOTAL)]

df_full = pd.DataFrame(data, index=postes_labels, columns=candidats_labels)

print(f"--- GÉNÉRATION DE LA MATRICE ({NB_POSTES_TOTAL}x{NB_CANDIDATS_TOTAL}) ---")
print("Exemple des 5 premières lignes/colonnes :")
print(df_full.iloc[:5, :5])
print("-" * 50)

# ---------------------------------------------------------
# 2. RÉSOLUTION PAR FORCE BRUTE (Sur échantillon réduit !)
# ---------------------------------------------------------
# ATTENTION : Impossible de faire 50 parmi 100 en brute force.
# Complexité factorielle trop grande. 
# On réduit à une "sous-équipe" de 5 postes parmi 12 candidats pour la démo.

nb_postes_demo = 5
nb_candidats_demo = 12

print(f"\n⚠️ ALERTE COMPLEXITÉ : La Force Brute ne peut pas gérer 50x100.")
print(f"Calcul sur un sous-ensemble : {nb_postes_demo} postes parmi {nb_candidats_demo} candidats...")

sub_df = df_full.iloc[:nb_postes_demo, :nb_candidats_demo]
candidats_demo = sub_df.columns.tolist()

start_time = time.time()

meilleur_score = 0
meilleure_compo = None
compteur = 0

# Test de toutes les permutations
for equipe in permutations(candidats_demo, nb_postes_demo):
    compteur += 1
    score_actuel = 0
    valid = True
    
    for i, candidat in enumerate(equipe):
        # On récupère la note
        score_actuel += sub_df.iloc[i][candidat]
    
    if score_actuel > meilleur_score:
        meilleur_score = score_actuel
        meilleure_compo = equipe

end_time = time.time()
duree = end_time - start_time

print(f"\n✅ Terminé en {duree:.4f} secondes.")
print(f"Combinaisons testées : {compteur}")
print(f"Score Max (sur l'échantillon) : {meilleur_score}")

# ---------------------------------------------------------
# 3. VISUALISATION (Heatmap)
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

# On affiche la heatmap des scores
sns.heatmap(sub_df, annot=True, fmt="d", cmap="YlGnBu", cbar=False)

# On entoure les choix de l'algo
# (Juste pour l'affichage, on doit retrouver les indices)
candidats_noms = sub_df.columns.tolist()
for i, candidat_choisi in enumerate(meilleure_compo):
    j = candidats_noms.index(candidat_choisi)
    # Rectangle rouge autour du choix
    plt.gca().add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=4))

plt.title(f"Résultat V3 (Brute Force) - Échantillon réduit\nScore: {meilleur_score}", fontsize=14)
plt.show()
