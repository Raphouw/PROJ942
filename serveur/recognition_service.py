import json
import os

import numpy as np

from pretraitement import OPTIONS_PAR_DEFAUT, preparer


def _charger_labels(chemin):
    brut = np.load(chemin, allow_pickle=True)
    if brut.dtype == object and brut.shape == ():
        brut = brut.item()
    elif brut.shape == ():
        brut = str(brut)
    if isinstance(brut, bytes):
        brut = brut.decode()
    if isinstance(brut, str):
        brut = json.loads(brut)
    if isinstance(brut, dict):
        return {int(k): str(v) for k, v in brut.items()}
    return {i: str(v) for i, v in enumerate(list(brut))}


def charger_modele(dossier):
    def charge(nom):
        return np.load(os.path.join(dossier, nom + ".npy"), allow_pickle=True)

    taille = int(charge("image_size").tolist()[0])
    options = dict(OPTIONS_PAR_DEFAUT)
    options["taille"] = taille
    chemin_options = os.path.join(dossier, "options.json")
    if os.path.exists(chemin_options):          # modeles crees par entrainer.py
        with open(chemin_options) as f:
            options.update(json.load(f))

    return {
        "W": charge("W").astype(np.float64),
        "mu": charge("mu").astype(np.float64),
        "Y_train": charge("Y_train").astype(np.float64),
        "y_train": charge("y_train").astype(np.int64),
        "labels": _charger_labels(os.path.join(dossier, "labels.npy")),
        "options": options,
    }


class RecognitionService:
    def __init__(self, dossier_modele, seuil=None):
        self.model = charger_modele(dossier_modele)
        self.seuil = seuil

    def analyser_image(self, image_bytes):
        x, _ = preparer(image_bytes, self.model["options"])
        Y = ((x.reshape(1, -1) - self.model["mu"]) @ self.model["W"]).reshape(-1)

        Y_train = self.model["Y_train"]
        y_train = self.model["y_train"]
        labels = self.model["labels"]

        distances = np.linalg.norm(Y_train - Y, axis=1)
        i = int(np.argmin(distances))
        distance_min = float(distances[i])
        classe = int(y_train[i])
        identite = "inconnu" if (self.seuil is not None and distance_min > self.seuil) \
            else labels[classe]

        comparaisons = []
        for c, nom in labels.items():
            d = distances[y_train == c]
            if len(d) > 0:
                comparaisons.append({"identite": nom, "distance": float(d.min())})
        comparaisons.sort(key=lambda r: r["distance"])
        poids = 1.0 / np.maximum([r["distance"] for r in comparaisons], 1e-9)
        scores = 100.0 * poids / poids.sum()
        for r, s in zip(comparaisons, scores):
            r["similarite"] = float(s)

        return {"identite": identite,
                "classe": None if identite == "inconnu" else classe,
                "distance": distance_min,
                "comparaisons": comparaisons}
