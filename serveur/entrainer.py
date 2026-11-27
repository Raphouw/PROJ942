"""Apprentissage ACP + evaluation.  Exemples :
  python3 entrainer.py Smile
  python3 entrainer.py Smile --egaliser --carre
  python3 entrainer.py Smile --recadrer --egaliser --retirer 3 --composantes 20
  python3 entrainer.py Smile --recadrer --egaliser --sortie acp_model_v2
"""
import argparse
import json
import os

import numpy as np

from pretraitement import preparer


def charger_dataset(dossier, options):
    X, y, labels, ignorees, sans_visage = [], [], {}, [], 0
    personnes = sorted(d for d in os.listdir(dossier)
                       if os.path.isdir(os.path.join(dossier, d)))
    for classe, nom in enumerate(personnes):
        labels[classe] = nom
        chemin = os.path.join(dossier, nom)
        for f in sorted(os.listdir(chemin)):
            if not f.lower().endswith((".jpg", ".jpeg", ".png", ".pgm")):
                continue
            try:
                v, detecte = preparer(os.path.join(chemin, f), options)
            except OSError:                 # fichier illisible / corrompu
                ignorees.append(nom + "/" + f)
                continue
            if detecte is False:
                sans_visage += 1
            X.append(v)
            y.append(classe)
    return np.array(X), np.array(y), labels, ignorees, sans_visage


def apprendre(X, retirer, composantes):
    mu = X.mean(axis=0)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    rang = int(np.count_nonzero(S > 1e-9 * S[0]))
    fin = rang if composantes <= 0 else min(retirer + composantes, rang)
    W = Vt[retirer:fin].T
    D = (S[retirer:fin] ** 2) / (len(X) - 1)
    return D, W, mu


def predire(Y_test, Y_train, y_train):
    d = np.linalg.norm(Y_train[None, :, :] - Y_test[:, None, :], axis=2)
    return y_train[np.argmin(d, axis=1)]


def evaluer(X, y, labels, args):
    rng = np.random.RandomState(args.graine)
    classes = sorted(labels)
    bons, total = 0, 0
    par_personne = {c: [0, 0] for c in classes}
    for _ in range(args.repetitions):
        train, test = [], []
        for c in classes:
            idx = np.where(y == c)[0]
            if len(idx) <= args.test + 1:
                train.extend(idx)          # trop peu d'images : pas de test
                continue
            idx = rng.permutation(idx)
            test.extend(idx[:args.test])
            train.extend(idx[args.test:])
        train, test = np.array(train), np.array(test)
        if len(test) == 0:
            return None
        D, W, mu = apprendre(X[train], args.retirer, args.composantes)
        Ytr, Yte = (X[train] - mu) @ W, (X[test] - mu) @ W
        pred = predire(Yte, Ytr, y[train])
        for p, v in zip(pred, y[test]):
            par_personne[v][0] += int(p == v)
            par_personne[v][1] += 1
            bons += int(p == v)
            total += 1
    return bons / total, par_personne


def sauver(sortie, D, W, mu, Y, y, labels, options):
    os.makedirs(sortie, exist_ok=True)
    np.save(os.path.join(sortie, "D.npy"), D)
    np.save(os.path.join(sortie, "W.npy"), W)
    np.save(os.path.join(sortie, "mu.npy"), mu)
    np.save(os.path.join(sortie, "Y_train.npy"), Y)
    np.save(os.path.join(sortie, "y_train.npy"), y.astype(np.int64))
    np.save(os.path.join(sortie, "labels.npy"), np.array(json.dumps(labels)))
    np.save(os.path.join(sortie, "image_size.npy"),
            np.array([options["taille"], options["taille"]], dtype=np.int64))
    with open(os.path.join(sortie, "options.json"), "w") as f:
        json.dump(options, f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dossier")
    p.add_argument("--taille", type=int, default=100)
    p.add_argument("--carre", action="store_true", help="carre central (garde les proportions)")
    p.add_argument("--recadrer", action="store_true", help="detecte et recadre le visage")
    p.add_argument("--egaliser", action="store_true", help="egalisation d'histogramme")
    p.add_argument("--retirer", type=int, default=0, help="nb de premieres composantes ignorees")
    p.add_argument("--composantes", type=int, default=0, help="nb de composantes gardees (0 = toutes)")
    p.add_argument("--test", type=int, default=3, help="images de test par personne")
    p.add_argument("--repetitions", type=int, default=30)
    p.add_argument("--graine", type=int, default=0)
    p.add_argument("--sortie", default=None, help="dossier du modele a enregistrer")
    args = p.parse_args()

    options = {"taille": args.taille, "recadrer": args.recadrer,
               "egaliser": args.egaliser, "carre": args.carre}
    X, y, labels, ignorees, sans_visage = charger_dataset(args.dossier, options)

    print("Options :", options, "| retirer", args.retirer, "| composantes", args.composantes)
    print("Images lues :", len(X), "| illisibles ignorees :", len(ignorees))
    for nom in ignorees:
        print("   ILLISIBLE :", nom)
    if args.recadrer:
        print("Images sans visage detecte (carre central utilise) :", sans_visage)
    for c in sorted(labels):
        print("   %-12s %d images" % (labels[c], int(np.sum(y == c))))

    res = evaluer(X, y, labels, args)
    if res is None:
        print("Pas assez d'images pour evaluer.")
    else:
        taux, par_personne = res
        print("\nTAUX DE RECONNAISSANCE (images de test hors apprentissage, moyenne sur %d tirages) : %.1f %%"
              % (args.repetitions, 100 * taux))
        for c in sorted(labels):
            b, t = par_personne[c]
            if t:
                print("   %-12s %5.1f %%" % (labels[c], 100.0 * b / t))

    if args.sortie:
        D, W, mu = apprendre(X, args.retirer, args.composantes)
        sauver(args.sortie, D, W, mu, (X - mu) @ W, y, labels, options)
        print("\nModele enregistre dans :", args.sortie)


if __name__ == "__main__":
    main()
