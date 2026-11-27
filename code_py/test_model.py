import argparse

import config
from recognition_service import RecognitionService

parser = argparse.ArgumentParser()
parser.add_argument("modele", nargs="?", help="nom du .npz (sinon le plus récent)")
parser.add_argument("--dataset", default=config.DATASET.name,
                    help="dataset_star ou dataset")
args = parser.parse_args()

# Charger le modèle entraîné
service = RecognitionService(config.choisir_modele(args.modele))

validation = config.RACINE / args.dataset / "validation"
dossiers = sorted(d for d in validation.iterdir() if d.is_dir())

# Vérifier que le modèle et la validation concernent les mêmes personnes
connus = set(service.model["labels"].values())
absents = [d.name for d in dossiers if d.name not in connus]

if absents:
    raise SystemExit(
        f"Personnes absentes du modèle : {', '.join(absents)}\n"
        "Vérifie le modèle choisi et l'option --dataset."
    )

total = 0
corrects = 0
ignorees = 0

# Parcourir les dossiers de validation
for dossier in dossiers:

    identite_attendue = dossier.name

    for photo in sorted(dossier.iterdir()):
        if photo.suffix.lower() not in (".jpg", ".jpeg", ".png", ".pgm"):
            continue

        # Lire et reconnaître la photo
        try:
            with open(photo, "rb") as fichier:
                resultat = service.analyser_image(fichier.read())
        except ValueError as erreur:
            ignorees += 1
            print(f"{photo.name} ignorée : {erreur}")
            continue

        prediction = resultat["identite"]
        distance = resultat["distance"]

        # Comparer avec le résultat attendu
        total += 1

        if prediction == identite_attendue:
            corrects += 1

        print(
            f"{photo.name} | "
            f"Attendu : {identite_attendue} | "
            f"Prédit : {prediction} | "
            f"Distance : {distance:.2f}"
        )

if total == 0:
    raise ValueError("Aucune photo de validation exploitable")

print("\n---------------------------")
print(f"Dataset : {args.dataset}")
print(f"Photos testées : {total}")
print(f"Photos ignorées : {ignorees}")
print(f"Reconnaissances correctes : {corrects}")
print(f"Précision : {100 * corrects / total:.2f}%")
