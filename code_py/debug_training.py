
import cv2

import config

from acp import read_images


# Charger exactement les images utilisées pour l'ACP
X, y, labels = read_images(
    config.TRAIN,
    sz=(100, 100)
)

sortie = config.RESULTATS / "visages_entrainement"
sortie.mkdir(parents=True, exist_ok=True)

compteurs = {}

for image, classe in zip(X, y):

    nom = labels[classe]

    dossier = sortie / nom
    dossier.mkdir(parents=True, exist_ok=True)

    compteurs[nom] = compteurs.get(nom, 0) + 1

    fichier = dossier / f"visage_{compteurs[nom]}.png"

    cv2.imwrite(str(fichier), image)

print("Visages utilisés pour l'ACP enregistrés.")
