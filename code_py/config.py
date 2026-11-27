from datetime import datetime
from pathlib import Path

# LE SEUL CHEMIN A MODIFIER : dossier qui contient code_py/, models/ et dataset_star/
# Par défaut : le dossier parent de code_py.
# Sinon, par ex. : RACINE = Path(r"T:\BRASSART\Proj942\!! Projet")
RACINE = Path(__file__).resolve().parent.parent

DATASET = RACINE / "dataset_star"
TRAIN = DATASET / "train"
VALIDATION = DATASET / "validation"
MODELS = RACINE / "models"
YUNET = MODELS / "face_detection_yunet_2022mar.onnx"
RESULTATS = RACINE / "resultats"
PHOTO_TEST = RACINE / "photo_test.jpg"
PHOTO_GROUPE = RACINE / "photos_groupes" / "groupe.jpg"


def nouveau_modele(num_components, dataset="dataset_star"):
    """Nom unique pour chaque entraînement."""
    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    return MODELS / f"acp_{dataset}_{date}_k{num_components}.npz"


def choisir_modele(nom=None):
    """Modèle demandé, sinon le plus récent."""
    if nom:
        chemin = MODELS / nom
    else:
        modeles = list(MODELS.glob("acp_*.npz"))
        if not modeles:
            raise FileNotFoundError(f"Aucun modèle dans {MODELS}")
        chemin = max(modeles, key=lambda p: p.stat().st_mtime)

    if not chemin.is_file():
        raise FileNotFoundError(f"Modèle introuvable : {chemin}")

    print(f"Modèle utilisé : {chemin.name}")
    return chemin
