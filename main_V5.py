import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from itertools import permutations
import time
import math

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
MAX_POSTES = 15  # On va jusqu'à 15 sur le graphique
SEUIL_REEL = 7  # Jusqu'à 7, on calcule vraiment (Mode V3). Après, on projette.

print(f"--- V5 (Mode V3) : COMPARAISON LOURDE ---")
print(f"Calcul RÉEL (Lecture + Addition + Comparaison) jusqu'à {SEUIL_REEL} postes.")
print("-" * 60)


# ---------------------------------------------------------
# 1. FONCTIONS
# ---------------------------------------------------------

def solve_simplex(n_postes):
    """ Algorithme Simplexe (Optimisé) """
    n_candidats = n_postes * 2
    c = np.random.randint(1, 21, size=n_postes * n_candidats) * -1

    A_eq = np.zeros((n_postes, n_postes * n_candidats))
    for i in range(n_postes):
        A_eq[i, i * n_candidats: (i + 1) * n_candidats] = 1
    b_eq = np.ones(n_postes)

    A_ub = np.zeros((n_candidats, n_postes * n_candidats))
    for j in range(n_candidats):
        for i in range(n_postes):
            A_ub[j, i * n_candidats + j] = 1
    b_ub = np.ones(n_candidats)

    start = time.time()
    linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=(0, 1), method='highs')
    return time.time() - start


def get_bruteforce_heavy(n_postes, time_per_op_ref=None):
    """
    Mode V3 : Fait vraiment le travail (Lecture mémoire + Calcul + Comparaison).
    """
    n_candidats = n_postes * 2
    nb_combinaisons = math.perm(n_candidats, n_postes)

    # CAS 1 : Calcul RÉEL (Comme V3)
    if n_postes <= SEUIL_REEL:
        # Génération des données pour faire le vrai travail
        data = np.random.randint(1, 21, size=(n_postes, n_candidats))

        start = time.time()
        meilleur_score = -1

        # La boucle lourde (identique à V3)
        for equipe in permutations(range(n_candidats), n_postes):
            score_actuel = 0
            # On parcourt chaque joueur de l'équipe pour sommer les notes
            for i, candidat_idx in enumerate(equipe):
                score_actuel += data[i, candidat_idx]  # Accès mémoire + addition

            if score_actuel > meilleur_score:
                meilleur_score = score_actuel

        duree = time.time() - start

        # On calcule le temps moyen par opération pour les futures projections
        new_time_per_op = duree / nb_combinaisons
        return duree, False, new_time_per_op

    # CAS 2 : Projection (Car trop long)
    else:
        # Temps = Nombre de combinaisons * Temps unitaire du dernier calcul réel
        temps_estime = nb_combinaisons * time_per_op_ref
        return temps_estime, True, time_per_op_ref


# ---------------------------------------------------------
# 2. EXÉCUTION
# ---------------------------------------------------------

x_axis = []
y_simplex = []
y_bruteforce = []
is_projected = []

# Stockera la vitesse du dernier calcul réel
last_time_per_op = 0

print(f"{'Postes':<10} | {'Combinaisons':<15} | {'Simplexe (s)':<15} | {'Brute Force (s)':<25} | {'Type'}")
print("-" * 85)

for n in range(2, MAX_POSTES + 1):
    n_candidats = n * 2
    nb_combi = math.perm(n_candidats, n)

    # 1. Simplexe
    t_sx = solve_simplex(n)

    # 2. Brute Force (Lourd)
    t_bf, projected, time_op = get_bruteforce_heavy(n, last_time_per_op)

    if not projected:
        last_time_per_op = time_op  # Mise à jour de la référence de vitesse

    # Stockage
    x_axis.append(n)
    y_simplex.append(t_sx)
    y_bruteforce.append(t_bf)
    is_projected.append(projected)

    # Affichage
    type_str = "PROJETÉ" if projected else "RÉEL (V3)"

    # Formatage du temps console
    if t_bf < 60:
        t_bf_str = f"{t_bf:.4f} s"
    elif t_bf < 3600:
        t_bf_str = f"{t_bf / 60:.1f} min"
    else:
        t_bf_str = f"{t_bf / 3600:.1f} h"

    print(f"{n:<10} | {nb_combi:<15} | {t_sx:.6f}        | {t_bf_str:<25} | {type_str}")

# ---------------------------------------------------------
# 3. GRAPHIQUE
# ---------------------------------------------------------
plt.figure(figsize=(12, 7))

# Courbe Simplexe
plt.plot(x_axis, y_simplex, 's-', color='green', linewidth=2, label='Simplexe (Polynomial)')

# Séparation Réel / Projeté
x_real = [x for i, x in enumerate(x_axis) if not is_projected[i]]
y_real = [y for i, y in enumerate(y_bruteforce) if not is_projected[i]]
x_proj = [x for i, x in enumerate(x_axis) if is_projected[i]]
y_proj = [y for i, y in enumerate(y_bruteforce) if is_projected[i]]

# Tracer Réel
plt.plot(x_real, y_real, 'o-', color='red', linewidth=2, label='Brute Force (Mesuré V3)')

# Tracer Projeté
if x_real and x_proj:
    plt.plot([x_real[-1], x_proj[0]], [y_real[-1], y_proj[0]], '--', color='red', alpha=0.5)
plt.plot(x_proj, y_proj, 'o--', color='red', alpha=0.5, label='Brute Force (Projeté)')

# Échelle Logarithmique
plt.yscale('log')

plt.title(
    f"Comparaison de Performance : Simplexe vs Brute Force (Logique V3)\nArrêt du calcul réel après {SEUIL_REEL} postes",
    fontsize=14)
plt.xlabel("Taille du problème (Nombre de Postes)")
plt.ylabel("Temps de résolution (Secondes) - Log Scale")
plt.grid(True, which="both", linestyle='--', alpha=0.4)
plt.legend()

# ---------------------------------------------------------
# 4. ANNOTATION FINALE UNIQUE
# ---------------------------------------------------------
val_finale = y_bruteforce[-1]

# Constantes de temps
ONE_YEAR = 3600 * 24 * 365
ONE_MONTH = 3600 * 24 * 30  # Approx 30 jours
ONE_DAY = 3600 * 24

# Choix de l'unité
if val_finale > ONE_YEAR:
    if val_finale / ONE_YEAR > 1000:
        txt_annot = f"{val_finale / ONE_YEAR:.1e} années"
    else:
        txt_annot = f"{val_finale / ONE_YEAR:.1f} années"
elif val_finale > ONE_MONTH:
    txt_annot = f"{val_finale / ONE_MONTH:.1f} mois"
elif val_finale > ONE_DAY:
    txt_annot = f"{val_finale / ONE_DAY:.1f} jours"
else:
    txt_annot = f"{val_finale:.0f} sec"

# Affichage de l'annotation unique
plt.annotate(f"Temps estimé pour {MAX_POSTES} postes :\n{txt_annot}",
             (x_axis[-1], y_bruteforce[-1]),
             xytext=(-100, 10), textcoords='offset points',
             arrowprops=dict(arrowstyle="->", color='black'),
             bbox=dict(boxstyle="round", fc="white"))

plt.tight_layout()
plt.show()