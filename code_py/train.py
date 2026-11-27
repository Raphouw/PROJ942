
import argparse
from pathlib import Path

import config

from acp import (
    read_images,
    asRowMatrix,
    ACP,
    project,
    save_model
)


def train_model(
    dataset_path=config.TRAIN,
    model_path=None,
    image_size=(100, 100),
    num_components=30
):
    """Entraîne l'ACP et sauvegarde le modèle."""

    # Sans chemin imposé, créer un nouveau fichier à chaque fois
    if model_path is None:
        model_path = config.nouveau_modele(
            num_components,
            Path(dataset_path).parent.name
        )

    model_path = Path(model_path)

    if model_path.exists():
        raise FileExistsError(f"Ce modèle existe déjà : {model_path}")

    # 1. Charger les photos et les labels
    X, y, labels = read_images(dataset_path, sz=image_size)

    if len(X) < 2:
        raise ValueError("Au moins deux images sont nécessaires")

    # 2. Transformer les images en matrice
    X_train = asRowMatrix(X)

    # 3. Calculer l'ACP
    D, W, mu = ACP(X_train, num_components)

    # 4. Projeter les photos d'entraînement
    Y_train = project(W, X_train, mu)

    # 5. Sauvegarder le modèle
    save_model(
        model_path,
        D, W, mu,
        Y_train, y, labels,
        image_size
    )

    print("Entraînement terminé")
    print(f"Nombre de photos : {len(X)}")
    print(f"Nombre de personnes : {len(labels)}")
    print(f"Composantes conservées : {W.shape[1]}")
    print(f"Modèle enregistré : {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=config.DATASET.name,
                        help="dataset_star ou dataset")
    parser.add_argument("--k", type=int, default=30,
                        help="nombre de composantes (0 = toutes)")
    args = parser.parse_args()

    train_model(
        dataset_path=config.RACINE / args.dataset / "train",
        num_components=args.k
    )
