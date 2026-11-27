import hashlib
import random
import shutil
import sys
from pathlib import Path

import config

RATIO_TRAIN = 0.75
# Mêmes formats que ceux lus par l'entraînement et le test
EXTENSIONS = {".jpg", ".jpeg", ".png", ".pgm"}
SORTIES = ("train", "validation")


def main():
    racine = Path(sys.argv[1] if len(sys.argv) > 1 else config.DATASET)
    if not racine.is_dir():
        sys.exit(f"Dossier introuvable : {racine}")

    source = racine / "all_pictures"
    if not source.is_dir():
        sys.exit(f"Dossier introuvable : {source}")

    # Évite qu'un 2e lancement mélange train et validation
    for nom in SORTIES:
        dossier = racine / nom
        if dossier.exists() and any(p.is_file() for p in dossier.rglob("*")):
            sys.exit(f"'{dossier}' contient déjà des fichiers, vide-le avant de relancer.")

    rng = random.Random(42)  # même découpage à chaque exécution

    for star in sorted(source.iterdir()):
        if not star.is_dir():
            continue

        # Ne garder qu'un exemplaire des fichiers strictement identiques
        uniques = {}
        doublons = 0

        for p in sorted(star.iterdir()):
            if not p.is_file() or p.suffix.lower() not in EXTENSIONS:
                continue

            empreinte = hashlib.sha256(p.read_bytes()).hexdigest()

            if empreinte in uniques:
                doublons += 1
            else:
                uniques[empreinte] = p

        images = list(uniques.values())
        rng.shuffle(images)
        coupe = round(len(images) * RATIO_TRAIN)

        for nom, lot in (("train", images[:coupe]), ("validation", images[coupe:])):
            destination = racine / nom / star.name
            destination.mkdir(parents=True, exist_ok=True)
            for image in lot:
                shutil.copy2(image, destination / image.name)

        print(
            f"{star.name} : {coupe} train / {len(images) - coupe} validation "
            f"({doublons} doublon(s) écarté(s))"
        )


if __name__ == "__main__":
    main()