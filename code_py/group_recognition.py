
import sys
from pathlib import Path

import cv2
import numpy as np

import config
from acp import load_model, project
from face_processing import detect_faces, preprocess_face


def compare_personnes(Y_test, model):
    """
    Compare un visage à toutes les personnes du modèle.

    Retourne les personnes classées par distance,
    avec un score de similarité relative.
    """

    Y_test = np.asarray(Y_test, dtype=np.float64).reshape(-1)

    Y_train = model["Y_train"]
    y_train = model["y_train"]
    labels = model["labels"]

    if Y_train.ndim != 2 or Y_train.shape[1] != Y_test.size:
        raise ValueError("Dimensions ACP incompatibles")

    if Y_train.shape[0] != len(y_train):
        raise ValueError("Modèle incohérent : labels manquants")

    # Distance à chaque photo d'entraînement
    distances = np.linalg.norm(
        Y_train - Y_test,
        axis=1
    )

    resultats = []

    for classe, nom in labels.items():

        # Distances des photos de cette personne
        distances_classe = distances[y_train == classe]

        if len(distances_classe) == 0:
            continue

        # Conserver sa photo la plus proche
        distance = float(np.min(distances_classe))

        resultats.append({
            "identite": nom,
            "distance": distance
        })

    if not resultats:
        raise ValueError("Aucune identité dans le modèle")

    # Trier par distance croissante
    resultats.sort(key=lambda r: r["distance"])

    # Similarités relatives (PAS des probabilités)
    d = np.array([r["distance"] for r in resultats])
    poids = 1.0 / np.maximum(d, 1e-9)
    scores = 100.0 * poids / poids.sum()

    for resultat, score in zip(resultats, scores):
        resultat["similarite"] = float(score)

    return resultats


def recognize_faces(image, model, seuil=None):
    """
    Détecte et reconnaît tous les visages d'une image.

    Retourne une liste de résultats.
    """

    rectangles = detect_faces(image)
    resultats = []

    for numero, rectangle in enumerate(rectangles, start=1):

        # Prétraitement identique à l'entraînement
        X_test = preprocess_face(
            image,
            rectangle,
            model["image_size"]
        )

        # Projection ACP
        Y_test = project(
            model["W"],
            X_test,
            model["mu"]
        )

        # Comparaison avec les personnes connues
        comparaisons = compare_personnes(Y_test, model)

        meilleur = comparaisons[0]
        distance = meilleur["distance"]

        # Rejet des inconnus si un seuil a été calibré
        inconnu = seuil is not None and distance > seuil

        identite = (
            "Inconnu" if inconnu
            else meilleur["identite"]
        )

        resultats.append({
            "numero": numero,
            "rectangle": rectangle,
            "identite": identite,
            "distance": distance,
            "similarite": meilleur["similarite"],
            "comparaisons": comparaisons
        })

    return resultats


def annotate_image(image, resultats):
    """
    Dessine les rectangles et les étiquettes
    sur une copie de la photo.
    """

    image_annotee = image.copy()

    for resultat in resultats:

        x, y, w, h = resultat["rectangle"]
        identite = resultat["identite"]
        similarite = resultat["similarite"]

        # Vert pour une identité proposée, orange pour inconnu
        couleur = (
            (0, 165, 255) if identite == "Inconnu"
            else (0, 200, 0)
        )

        # Rectangle autour du visage
        cv2.rectangle(
            image_annotee,
            (x, y),
            (x + w, y + h),
            couleur,
            2
        )

        # Étiquette : score relatif et non probabilité
        texte = f"{identite} - {similarite:.1f}% rel."

        police = cv2.FONT_HERSHEY_SIMPLEX
        taille = 0.55
        epaisseur = 2

        (largeur_texte, hauteur_texte), baseline = (
            cv2.getTextSize(texte, police, taille, epaisseur)
        )

        # Placer l'étiquette au-dessus si possible
        haut = y - hauteur_texte - baseline - 10

        if haut < 0:
            haut = y + h + 2

        # Garder l'étiquette dans l'image
        haut = max(
            0,
            min(haut, image_annotee.shape[0] - hauteur_texte - baseline - 8)
        )
        gauche = max(
            0,
            min(x, max(0, image_annotee.shape[1] - largeur_texte - 10))
        )

        # Fond de l'étiquette
        cv2.rectangle(
            image_annotee,
            (gauche, haut),
            (min(gauche + largeur_texte + 10, image_annotee.shape[1] - 1),
             haut + hauteur_texte + baseline + 8),
            couleur,
            -1
        )

        # Texte en blanc
        cv2.putText(
            image_annotee,
            texte,
            (gauche + 5, haut + hauteur_texte + 3),
            police,
            taille,
            (255, 255, 255),
            epaisseur,
            cv2.LINE_AA
        )

    return image_annotee


def process_group_photo(
    input_path,
    output_path,
    model_path=None,
    seuil=None
):
    """
    Analyse une photo de groupe et sauvegarde
    une image annotée.
    """

    # Charger le modèle ACP
    if model_path is None:
        model_path = config.choisir_modele()

    model = load_model(model_path)

    # Charger la photo
    image = cv2.imread(str(input_path))

    if image is None:
        raise FileNotFoundError(
            f"Impossible d'ouvrir : {input_path}"
        )

    # Détecter et reconnaître les visages
    resultats = recognize_faces(image, model, seuil)

    # Dessiner les rectangles et les labels
    image_annotee = annotate_image(image, resultats)

    # Créer le dossier de sortie
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Enregistrer la photo
    if not cv2.imwrite(str(output_path), image_annotee):
        raise IOError("Impossible d'enregistrer l'image annotée")

    # Afficher les résultats dans le terminal
    print(f"\nVisages détectés : {len(resultats)}")

    for resultat in resultats:

        print(
            f"\nVisage {resultat['numero']} : "
            f"{resultat['identite']}"
        )

        for personne in resultat["comparaisons"]:
            print(
                f"  {personne['identite']:<15} "
                f"Distance : {personne['distance']:>10.2f} "
                f"Score relatif : {personne['similarite']:>6.2f}%"
            )

    print(f"\nImage enregistrée : {output_path}")

    return resultats


if __name__ == "__main__":

    modele = config.choisir_modele(
        sys.argv[1] if len(sys.argv) > 1 else None
    )

    process_group_photo(
        input_path=config.PHOTO_GROUPE,
        output_path=config.RESULTATS / f"groupe_{modele.stem}.jpg",
        model_path=modele,
        seuil=None
    )
