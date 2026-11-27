"""
Visualisation des résultats de l'ACP.

    python code_py/visualiser.py MODELE.npz
    python code_py/visualiser.py MODELE.npz --photo chemin/vers/photo.jpg
    python code_py/visualiser.py MODELE.npz --dataset dataset --fiches 0

Sorties : resultats/visualisation/NOM_DU_MODELE/
"""
import argparse
import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # enregistre les figures sans ouvrir de fenêtre
import matplotlib.pyplot as plt
import numpy as np

import config
from acp import load_model, project, reconstruct, compare_all
from face_processing import (
    detect_faces_detailed,
    suppress_contained_detections,
    select_main_face,
    extract_face,
)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".pgm"}
BLEU, GRIS, VERT, ROUGE = "#2a78d6", "#c3c2b7", "#008300", "#c62f2f"


def enregistrer(fig, chemin):
    chemin.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chemin, dpi=130, bbox_inches="tight")
    plt.close(fig)


def analyser_photo(chemin, model):
    """Détecte, projette et compare une photo. None si aucun visage."""
    image = cv2.imread(str(chemin))
    if image is None:
        return None

    detections = suppress_contained_detections(detect_faces_detailed(image))
    if not detections:
        return None

    rect = select_main_face(detections, image.shape)
    visage = extract_face(image, rect, model["image_size"])
    y = project(model["W"], visage.reshape(1, -1).astype(float), model["mu"])[0]

    return {
        "chemin": chemin,
        "image": image,
        "rect": rect,
        "visage": visage,
        "y": y,
    }


def charger(dossier, model):
    resultats, ignorees = [], 0

    for personne in sorted(d for d in dossier.iterdir() if d.is_dir()):
        for photo in sorted(personne.iterdir()):
            if photo.suffix.lower() not in EXTENSIONS:
                continue

            r = analyser_photo(photo, model)
            if r is None:
                ignorees += 1
                continue

            r["attendu"] = personne.name
            # On ne garde pas la photo entière en mémoire
            del r["image"]
            resultats.append(r)

        print(f"  {personne.name}")

    return resultats, ignorees


def classer(resultats, model, entrainement):
    """
    Plus proche voisin. Sur le train, chaque photo se retrouverait
    elle-même (distance 0, 100 %) : on exclut donc la photo identique
    ("leave-one-out").
    """
    noms = model["labels"]

    for r in resultats:
        distances = np.linalg.norm(model["Y_train"] - r["y"], axis=1)

        if entrainement:
            i = int(np.argmin(distances))
            if distances[i] < 1e-6:
                distances[i] = np.inf

        i = int(np.argmin(distances))
        r["predit"] = noms[int(model["y_train"][i])]
        r["distance"] = float(distances[i])


# ---------------------------------------------------------------- Vues globales

def visage_moyen_et_composantes(model, sortie, nombre=15):
    taille = model["image_size"][::-1]
    k = min(nombre, model["W"].shape[1])
    colonnes = 4
    lignes = int(np.ceil((k + 1) / colonnes))
    fig, axes = plt.subplots(lignes, colonnes, figsize=(10, 2.7 * lignes))

    for ax in axes.flat:
        ax.axis("off")

    axes.flat[0].imshow(model["mu"].reshape(taille), cmap="gray")
    axes.flat[0].set_title("Visage moyen")

    total = model["D"].sum()
    for i in range(k):
        ax = axes.flat[i + 1]
        ax.imshow(model["W"][:, i].reshape(taille), cmap="gray")
        part = 100 * model["D"][i] / total
        ax.set_title(f"CP{i + 1} ({part:.0f} % des k axes)")

    enregistrer(fig, sortie / "1_composantes.png")


def composantes_par_personne(model, sortie):
    """Position moyenne de chaque personne sur chaque axe."""
    noms = [model["labels"][c] for c in sorted(model["labels"])]
    moyennes = np.array([
        model["Y_train"][model["y_train"] == c].mean(axis=0)
        for c in sorted(model["labels"])
    ])
    # Chaque axe est divisé par son écart-type pour être comparable
    z = moyennes / np.sqrt(model["D"])
    k = z.shape[1]
    borne = np.abs(z).max()

    fig, ax = plt.subplots(figsize=(max(9, 0.33 * k), 0.42 * len(noms) + 1.5))
    im = ax.imshow(z, cmap="RdBu_r", vmin=-borne, vmax=borne, aspect="auto")
    ax.set_yticks(range(len(noms)), noms)
    ax.set_xticks(range(k), range(1, k + 1), fontsize=7)
    ax.set_xlabel("Composante principale")
    ax.set_title("Coordonnée moyenne par personne (en écarts-types)")
    fig.colorbar(im, ax=ax)
    enregistrer(fig, sortie / "2_composantes_par_personne.png")

    # Visage moyen de chaque personne, reconstruit avec les k composantes
    colonnes = 6
    lignes = int(np.ceil(len(noms) / colonnes))
    fig, axes = plt.subplots(lignes, colonnes, figsize=(13, 2.5 * lignes),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    for ax, nom, y in zip(axes.flat, noms, moyennes):
        x = reconstruct(model["W"], y, model["mu"])
        ax.imshow(x.reshape(model["image_size"][::-1]), cmap="gray",
                  vmin=0, vmax=255)
        ax.set_title(nom, fontsize=10)

    enregistrer(fig, sortie / "3_visage_moyen_par_personne.png")


def nuages(model, validation, sortie):
    """Un graphique par personne dans le plan CP1 / CP2."""
    if model["Y_train"].shape[1] < 2:
        return

    classes = sorted(model["labels"])
    colonnes = 5
    lignes = int(np.ceil(len(classes) / colonnes))
    fig, axes = plt.subplots(lignes, colonnes, figsize=(15, 3 * lignes),
                             sharex=True, sharey=True, squeeze=False)

    for ax in axes.flat[len(classes):]:
        ax.axis("off")

    for ax, c in zip(axes.flat, classes):
        nom = model["labels"][c]
        ax.scatter(*model["Y_train"][:, :2].T, s=4, color=GRIS, alpha=0.4)
        pts = model["Y_train"][model["y_train"] == c, :2]
        ax.scatter(*pts.T, s=9, color=BLEU, label="train")

        pts = np.array([r["y"][:2] for r in validation if r["attendu"] == nom])
        if len(pts):
            ax.scatter(*pts.T, s=28, marker="x", color="black",
                       label="validation")

        ax.set_title(nom, fontsize=10)

    axes.flat[0].legend(fontsize=8)
    fig.supxlabel("CP1")
    fig.supylabel("CP2")
    enregistrer(fig, sortie / "4_nuages_CP1_CP2.png")


def confusion(resultats, model, titre, chemin):
    noms = [model["labels"][c] for c in sorted(model["labels"])]
    index = {n: i for i, n in enumerate(noms)}
    m = np.zeros((len(noms), len(noms)), dtype=int)

    for r in resultats:
        m[index[r["attendu"]], index[r["predit"]]] += 1

    pourcent = 100 * m / np.maximum(m.sum(axis=1, keepdims=True), 1)
    precision = 100 * np.trace(m) / max(m.sum(), 1)

    cote = max(6, 0.6 * len(noms))
    fig, ax = plt.subplots(figsize=(cote + 1, cote))
    im = ax.imshow(pourcent, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(noms)), noms, rotation=90)
    ax.set_yticks(range(len(noms)), noms)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Attendu")
    ax.set_title(f"{titre} : {precision:.1f} % ({np.trace(m)}/{m.sum()})")

    for i in range(len(noms)):
        for j in range(len(noms)):
            if m[i, j]:
                ax.text(j, i, m[i, j], ha="center", va="center", fontsize=7,
                        color="white" if pourcent[i, j] > 55 else "black")

    fig.colorbar(im, ax=ax, label="% de la ligne", shrink=0.8)
    enregistrer(fig, chemin)


def planches(resultats, dossier, par_page=40):
    """Tous les visages d'une personne, avec la prédiction de chacun."""
    for nom in sorted({r["attendu"] for r in resultats}):
        liste = [r for r in resultats if r["attendu"] == nom]
        bons = sum(r["predit"] == nom for r in liste)

        for page, debut in enumerate(range(0, len(liste), par_page), start=1):
            lot = liste[debut:debut + par_page]
            lignes = int(np.ceil(len(lot) / 8))
            fig, axes = plt.subplots(lignes, 8, figsize=(14, 2.2 * lignes + 0.6),
                                     squeeze=False)
            for ax in axes.flat:
                ax.axis("off")

            for ax, r in zip(axes.flat, lot):
                ok = r["predit"] == nom
                ax.imshow(r["visage"], cmap="gray", vmin=0, vmax=255)
                ax.set_title(
                    f"{'OK' if ok else 'ERREUR'} : {r['predit']}\n"
                    f"d = {r['distance']:.0f}",
                    fontsize=8, color=VERT if ok else ROUGE
                )

            fig.suptitle(f"{nom} : {bons}/{len(liste)} corrects")
            fig.tight_layout()
            enregistrer(fig, dossier / f"{nom}_{page:02d}.png")


# ---------------------------------------------------------------- Fiche photo

def fiche(chemin, model, sortie, attendu=None):
    r = analyser_photo(chemin, model)
    if r is None:
        print(f"Aucun visage exploitable : {chemin}")
        return

    comparaisons = compare_all(r["y"], model["Y_train"], model["y_train"],
                               model["labels"])
    meilleur = comparaisons[0]["identite"]
    noms = [c["identite"] for c in comparaisons]
    couleurs = [BLEU if n == meilleur else GRIS for n in noms]

    fig = plt.figure(figsize=(15, 10))
    grille = fig.add_gridspec(3, 3, height_ratios=[1.1, 1.5, 0.9],
                              hspace=0.55, wspace=0.5)

    image = r["image"].copy()
    x, y, w, h = r["rect"]
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 200, 0),
                  max(2, image.shape[1] // 200))
    rec = reconstruct(model["W"], r["y"], model["mu"])

    vues = [
        (cv2.cvtColor(image, cv2.COLOR_BGR2RGB), "Détection"),
        (r["visage"], "Entrée de l'ACP"),
        (rec.reshape(model["image_size"][::-1]),
         f"Reconstruction ({model['W'].shape[1]} composantes)"),
    ]
    for i, (img, titre) in enumerate(vues):
        ax = fig.add_subplot(grille[0, i])
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.set_title(titre)
        ax.axis("off")

    panneaux = [
        ("distance", "Distance minimale (plus petit = plus proche)", "{:.0f}"),
        ("similarite", "Similarité relative en % (pas une probabilité)",
         "{:.1f}"),
    ]
    for i, (cle, titre, forme) in enumerate(panneaux):
        ax = fig.add_subplot(grille[1, 0] if i == 0 else grille[1, 2])
        valeurs = [c[cle] for c in comparaisons]
        barres = ax.barh(noms, valeurs, color=couleurs)
        ax.bar_label(barres, labels=[forme.format(v) for v in valeurs],
                     padding=3, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(titre, fontsize=10)
        ax.set_xlim(0, max(valeurs) * 1.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)

    ax = fig.add_subplot(grille[2, :])
    z = r["y"] / np.sqrt(model["D"])
    ax.bar(range(1, len(z) + 1), z, color=BLEU)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Composante principale")
    ax.set_ylabel("Écarts-types")
    ax.set_title("Coordonnées de cette photo dans l'espace ACP")
    ax.spines[["top", "right"]].set_visible(False)

    titre = f"{chemin.name} | Prédit : {meilleur}"
    if attendu:
        titre += f" | Attendu : {attendu}"
    fig.suptitle(titre, fontsize=14)

    dossier = sortie / "fiches"
    prefixe = f"{attendu}_" if attendu else ""
    enregistrer(fig, dossier / f"{prefixe}{chemin.stem}.png")


def exporter_csv(validation, model, chemin):
    """Distances et similarités de chaque photo de test, pour chaque classe."""
    with open(chemin, "w", newline="", encoding="utf-8-sig") as f:
        sortie = csv.writer(f, delimiter=";")
        sortie.writerow(["photo", "attendu", "predit", "classe_comparee",
                         "distance", "similarite_pourcent"])

        for r in validation:
            for c in compare_all(r["y"], model["Y_train"], model["y_train"],
                                 model["labels"]):
                sortie.writerow([
                    r["chemin"].name, r["attendu"], r["predit"], c["identite"],
                    f"{c['distance']:.2f}".replace(".", ","),
                    f"{c['similarite']:.3f}".replace(".", ","),
                ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("modele", nargs="?", help="nom du .npz")
    parser.add_argument("--dataset", default=config.DATASET.name)
    parser.add_argument("--photo", type=Path, help="analyser une seule photo")
    parser.add_argument("--fiches", type=int, default=2,
                        help="fiches détaillées par personne (-1 = toutes)")
    args = parser.parse_args()

    chemin_modele = config.choisir_modele(args.modele)
    model = load_model(chemin_modele)
    sortie = config.RESULTATS / "visualisation" / chemin_modele.stem

    if args.photo:
        fiche(args.photo, model, sortie)
        print(f"Résultats : {sortie}")
        return

    base = config.RACINE / args.dataset
    connus = set(model["labels"].values())
    for split in ("train", "validation"):
        absents = [d.name for d in (base / split).iterdir()
                   if d.is_dir() and d.name not in connus]
        if absents:
            raise SystemExit(f"Personnes absentes du modèle : {absents}")

    visage_moyen_et_composantes(model, sortie)
    composantes_par_personne(model, sortie)

    jeux = {}
    for split in ("train", "validation"):
        print(f"Lecture de {split}...")
        resultats, ignorees = charger(base / split, model)
        if not resultats:
            raise SystemExit(f"Aucun visage exploitable dans {split}")

        classer(resultats, model, entrainement=(split == "train"))
        print(f"{split} : {len(resultats)} visages, {ignorees} ignorées")

        titre = ("Train (leave-one-out)" if split == "train"
                 else "Validation")
        confusion(resultats, model, titre, sortie / f"5_confusion_{split}.png")
        planches(resultats, sortie / f"planches_{split}")
        jeux[split] = resultats

    validation = jeux["validation"]
    nuages(model, validation, sortie)
    exporter_csv(validation, model, sortie / "6_distances_validation.csv")

    compteur = {}
    for r in validation:
        n = compteur.get(r["attendu"], 0)
        if args.fiches < 0 or n < args.fiches:
            fiche(r["chemin"], model, sortie, r["attendu"])
            compteur[r["attendu"]] = n + 1

    print(f"Terminé. Résultats : {sortie}")


if __name__ == "__main__":
    main()
