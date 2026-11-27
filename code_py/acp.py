import sys
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import json
from pathlib import Path
from face_processing import (
    detect_faces_detailed,
    suppress_contained_detections,
    select_main_face,
    extract_face
)

"""
FONCTION POUR CHARGER LES PHOTOS ET LEUR ATTRIBUER UN LABEFL EN FONCTION 
D'UN PATH ET D'UN REDIMENSIONNEMENT (sz) NON OBLIGATOIRE

X = tableau de photos
y = numéro de label associé au photo
labels = compteur de labels
"""

from pathlib import Path

import numpy as np
from PIL import Image



def read_images(path, sz=(100, 100)):
    """
    Charge les photos d'entraînement.

    Détecte le visage dans chaque photo et le prépare
    pour l'ACP.

    X      : liste des images préparées
    y      : labels numériques
    labels : correspondance classe -> nom
    """

    X = []
    y = []
    labels = {}

    dataset = Path(path)

    if not dataset.is_dir():
        raise FileNotFoundError(f"Dataset introuvable : {dataset}")

    dossiers = sorted(
        dossier for dossier in dataset.iterdir()
        if dossier.is_dir()
    )

    for dossier in dossiers:

        # Utiliser uniquement les personnes ayant des photos valides
        images_personne = []

        for fichier in sorted(dossier.iterdir()):

            if fichier.suffix.lower() not in (
                ".jpg", ".jpeg", ".png", ".pgm"
            ):
                continue

            # Lire l'image avec OpenCV
            image = cv2.imread(str(fichier))

            if image is None:
                print(f"Image illisible : {fichier}")
                continue

            # Détecter les visages
            # 1. Détecter tous les visages
            detections = detect_faces_detailed(image)

            # 2. Supprimer les petites détections incluses
            detections = suppress_contained_detections(detections)

            # 3. Vérifier qu'au moins un visage est détecté
            if len(detections) == 0:
                print(f"Image ignorée : {fichier.name} (0 visage détecté)")
                continue

            # 4. Sélectionner le visage principal
            rectangle = select_main_face(detections, image.shape)

            if rectangle is None:
                print(f"Image ignorée : {fichier.name} (aucun visage principal)")
                continue

            # 5. Extraire et préparer le visage
            visage = extract_face(image, rectangle, sz)

            # 6. Ajouter le visage au dataset
            images_personne.append(visage)

            if len(detections) > 1:
                print(
                    f"{fichier.name} : {len(detections)} détections "
                    f"-> visage principal sélectionné"
                )

        # Ne créer un label que pour une personne ayant des images
        if not images_personne:
            print(f"Aucune image valide pour {dossier.name}")
            continue

        classe = len(labels)
        labels[classe] = dossier.name

        for visage in images_personne:
            X.append(visage)
            y.append(classe)

        print(
            f"{dossier.name} : "
            f"{len(images_personne)} visage(s) chargé(s)"
        )

    return X, y, labels


"""
Convertit x images (n lignes * p colonnes) en une seule matrice
de x lignes de n*p.

"""

def asRowMatrix(X):
    if len(X) == 0:
        return np.array([])

    return np.stack([
        image.reshape(-1) for image in X
    ], axis=0)

"""
Convertit x images (n lignes * p colonnes) en une seule matrice
de x colonnes de n*p.

"""
 
def asColumnMatrix(X):
    if len(X) == 0:
        return np.array([])
    return asRowMatrix(X).T


"""
Fonction principale :
  - X : Matrice contenant les images (1 images par lignes)
  - num_components : Nombre de composantes à garder.

1. Récupérer les dimensions de la matrice contenant les visages.
2. Calculer le visage moyen en faisant la moyenne de chaque pixel.
3. Centrer les images en soustrayant ce visage moyen.
4. Calculer et classer les directions principales grâce aux vecteurs propres ou à la SVD.
5. Conserver les meilleures composantes pour réduire la dimension des images.

"""
def ACP(X, num_components=0):
    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError("X doit être une matrice 2D")

    n, d = X.shape

    if n < 2 or d == 0:
        raise ValueError("Au moins 2 images et 1 pixel sont requis")

    # 1. Calcul du visage moyen
    mu = X.mean(axis=0)

    # 2. Centrage des données
    X_centre = X - mu

    # 3. Décomposition en valeurs singulières
    U, S, Vt = np.linalg.svd(
        X_centre,
        full_matrices=False
    )

    # 4. Filtrer les valeurs singulières nulles
    tol = np.finfo(float).eps * max(n, d) * S[0]
    rang = np.count_nonzero(S > tol)

    if rang == 0:
        raise ValueError("Les images ne présentent aucune variance exploitable")

    # 5. Choisir le nombre de composantes
    if num_components <= 0:
        num_components = rang
    else:
        num_components = min(num_components, rang)

    # 6. Sélection des composantes
    W = Vt[:num_components].T

    # 7. Valeurs propres de la covariance
    D = S[:num_components] ** 2 / (n - 1)

    return D, W, mu

"""


"""

def project(W, X, mu=None):
    if mu is not None:
        X = X - mu

    return X @ W
    
"""


"""

def reconstruct(W, Y, mu=None):
    X_reconstruit = Y @ W.T

    if mu is not None:
        X_reconstruit = X_reconstruit + mu

    return X_reconstruit


def euclidean_distance(p, q):
    p = np.asarray(p, dtype=np.float64).flatten()
    q = np.asarray(q, dtype=np.float64).flatten()

    if p.shape != q.shape:
        raise ValueError("Les deux vecteurs doivent avoir la même dimension")

    return np.linalg.norm(p - q)





def predict(Y_test, Y_train, y_train, labels, seuil=None):
    """
    Recherche le visage connu le plus proche.

    Y_test  : projection du visage à identifier
    Y_train : projections des visages d'entraînement
    y_train : classes des visages d'entraînement
    labels  : dictionnaire {classe: nom}
    seuil   : distance maximale acceptée (facultatif)

    Retourne un dictionnaire contenant le résultat.
    """

    Y_test = np.asarray(Y_test, dtype=np.float64).reshape(-1)
    Y_train = np.asarray(Y_train, dtype=np.float64)
    y_train = np.asarray(y_train)

    if Y_train.ndim != 2 or len(Y_train) == 0:
        raise ValueError("Projections d'entraînement invalides")

    if Y_train.shape[0] != len(y_train):
        raise ValueError("Nombre de projections et labels différent")

    if Y_train.shape[1] != Y_test.size:
        raise ValueError("Dimensions des projections incompatibles")

    # Calculer les distances avec tous les visages connus
    distances = np.linalg.norm(Y_train - Y_test, axis=1)

    # Trouver la plus petite distance
    indice = int(np.argmin(distances))
    distance_min = float(distances[indice])

    # Classe correspondant au visage le plus proche
    classe = int(y_train[indice])

    # Vérifier le seuil de reconnaissance
    if seuil is not None and distance_min > seuil:
        return {
            "identite": "inconnu",
            "classe": None,
            "distance": distance_min
        }

    return {
        "identite": labels[classe],
        "classe": classe,
        "distance": distance_min
    }



def compare_all(Y_test, Y_train, y_train, labels):
    """
    Compare une photo avec toutes les personnes connues.

    Retourne :
      - la distance minimale par personne
      - un score de similarité relative en %
    """

    Y_test = np.asarray(Y_test, dtype=np.float64).reshape(-1)
    Y_train = np.asarray(Y_train, dtype=np.float64)
    y_train = np.asarray(y_train)

    if Y_train.ndim != 2 or Y_train.shape[0] == 0:
        raise ValueError("Projections d'entraînement invalides")

    if Y_train.shape != (len(y_train), Y_test.size):
        raise ValueError("Dimensions incompatibles")

    # 1. Distances avec toutes les photos d'entraînement
    distances = np.linalg.norm(
        Y_train - Y_test,
        axis=1
    )

    # 2. Conserver la plus petite distance par personne
    resultats = []

    for classe, nom in labels.items():
        distances_personne = distances[y_train == classe]

        if len(distances_personne) == 0:
            continue

        distance_min = float(np.min(distances_personne))

        resultats.append({
            "identite": nom,
            "distance": distance_min
        })

    if not resultats:
        raise ValueError("Aucune personne à comparer")

    # 3. Trier de la personne la plus proche
    # à la personne la plus éloignée
    resultats.sort(key=lambda x: x["distance"])

    # 4. Transformer les distances en scores relatifs
    # Plus la distance est petite, plus le score est grand.
    distances_classes = np.array(
        [r["distance"] for r in resultats]
    )

    # Echelle indicative :
    # évite la division par zéro, même pour une photo identique
    epsilon = 1e-9

    poids = 1.0 / np.maximum(distances_classes, epsilon)
    scores = 100.0 * poids / poids.sum()

    for resultat, score in zip(resultats, scores):
        resultat["similarite"] = float(score)

    return resultats


def save_model(path, D, W, mu, Y_train, y_train, labels, image_size):
    """Sauvegarde les données nécessaires à la reconnaissance."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        path,
        D=D,
        W=W,
        mu=mu,
        Y_train=Y_train,
        y_train=np.asarray(y_train, dtype=np.int64),
        labels=json.dumps(labels),
        image_size=np.asarray(image_size, dtype=np.int64)
    )


def load_model(path):
    """Charge un modèle ACP enregistré sur disque."""

    with np.load(path, allow_pickle=False) as data:

        labels = {
            int(k): v
            for k, v in json.loads(str(data["labels"])).items()
        }

        return {
            "D": data["D"],
            "W": data["W"],
            "mu": data["mu"],
            "Y_train": data["Y_train"],
            "y_train": data["y_train"],
            "labels": labels,
            "image_size": tuple(data["image_size"].tolist())
        }
