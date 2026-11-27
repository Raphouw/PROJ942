
import sys

import config
from recognition_service import RecognitionService

service = RecognitionService(
    config.choisir_modele(sys.argv[1] if len(sys.argv) > 1 else None)
)

photo = config.PHOTO_TEST

with open(photo, "rb") as fichier:
    resultat = service.analyser_image(fichier.read())

print("\n========== RECONNAISSANCE ACP ==========")
print("Photo :", photo.name)
print("Identité prédite :", resultat["identite"])

print("\nComparaison avec toutes les personnes :")
print("-" * 55)

for personne in resultat["comparaisons"]:
    print(
        f"{personne['identite']:<15}"
        f"Distance : {personne['distance']:>10.2f}  "
        f"Score : {personne['similarite']:>6.2f}%"
    )

print("-" * 55)
print("Note : les scores sont des similarités relatives,")
print("pas des probabilités de reconnaissance.")
