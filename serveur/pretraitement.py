"""Pretraitement UNIQUE, utilise a l'apprentissage ET a la reconnaissance."""
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps

try:
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    LANCZOS = Image.LANCZOS

OPTIONS_PAR_DEFAUT = {"taille": 100, "recadrer": False, "egaliser": False, "carre": False}

_cascade = None


def _detecteur():
    global _cascade
    if _cascade is None:
        import cv2
        _cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if _cascade.empty():
            raise RuntimeError("Detecteur de visages OpenCV introuvable "
                               "(installer : pip3 install --user opencv-python-headless)")
    return _cascade


def _carre_central(im):
    w, h = im.size
    c = min(w, h)
    x, y = (w - c) // 2, (h - c) // 2
    return im.crop((x, y, x + c, y + c))


def _recadrer_visage(im):
    """Recadre sur le plus grand visage detecte, sinon carre central."""
    # detection sur une version reduite (plus rapide)
    echelle = 600.0 / max(im.size) if max(im.size) > 600 else 1.0
    petit = np.asarray(im.resize((int(im.width * echelle), int(im.height * echelle))),
                       dtype=np.uint8)
    visages = _detecteur().detectMultiScale(petit, 1.1, 5, minSize=(40, 40))
    if len(visages) == 0:
        return _carre_central(im), False
    x, y, w, h = max(visages, key=lambda f: f[2] * f[3])
    x, y, w, h = [int(v / echelle) for v in (x, y, w, h)]
    m = int(0.1 * w)
    gauche, haut = max(0, x - m), max(0, y - m)
    droite, bas = min(im.width, x + w + m), min(im.height, y + h + m)
    return im.crop((gauche, haut, droite, bas)), True


def preparer(source, options=None):
    """source : octets JPEG, chemin de fichier ou image PIL.
    Retourne (vecteur ligne float64, visage_detecte)."""
    opt = dict(OPTIONS_PAR_DEFAUT)
    if options:
        opt.update(options)

    if isinstance(source, Image.Image):
        im = source
    elif isinstance(source, (bytes, bytearray)):
        im = Image.open(BytesIO(source))
    else:
        im = Image.open(source)

    im.load()
    im = ImageOps.exif_transpose(im).convert("L")

    detecte = None
    if opt["recadrer"]:
        im, detecte = _recadrer_visage(im)
    elif opt["carre"]:
        im = _carre_central(im)

    t = int(opt["taille"])
    im = im.resize((t, t), LANCZOS)
    if opt["egaliser"]:
        im = ImageOps.equalize(im)
    return np.asarray(im, dtype=np.float64).reshape(-1), detecte
