import cv2
import numpy as np

import config

# Modèle YuNet compatible avec ton OpenCV 4.7.0
MODEL_PATH = config.YUNET

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Modèle YuNet introuvable : {MODEL_PATH}")


def detect_faces_detailed(image):
    """
    Détecte les visages avec YuNet.

    Retourne une liste de dictionnaires :
    [
        {
            "rect": (x, y, w, h),
            "score": confiance
        },
        ...
    ]
    """

    if image is None or image.size == 0:
        raise ValueError("Image vide ou invalide")

    if image.ndim == 2:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 3:
        image_bgr = image
    else:
        raise ValueError("Format d'image non pris en charge")

    hauteur, largeur = image_bgr.shape[:2]

    # Redimensionnement pour la détection
    taille_max = 960
    facteur = min(1.0, taille_max / max(largeur, hauteur))

    nouvelle_largeur = max(1, round(largeur * facteur))
    nouvelle_hauteur = max(1, round(hauteur * facteur))

    image_reduite = cv2.resize(
        image_bgr,
        (nouvelle_largeur, nouvelle_hauteur),
        interpolation=cv2.INTER_AREA
    )

    facteur_x = nouvelle_largeur / largeur
    facteur_y = nouvelle_hauteur / hauteur

    # Compléter à un multiple de 32
    largeur_reseau = ((nouvelle_largeur + 31) // 32) * 32
    hauteur_reseau = ((nouvelle_hauteur + 31) // 32) * 32

    image_preparee = cv2.copyMakeBorder(
        image_reduite,
        0,
        hauteur_reseau - nouvelle_hauteur,
        0,
        largeur_reseau - nouvelle_largeur,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    detecteur = cv2.FaceDetectorYN.create(
        str(MODEL_PATH),
        "",
        (largeur_reseau, hauteur_reseau),
        0.7,   # score threshold
        0.3,   # nms threshold
        5000   # top_k
    )

    _, visages = detecteur.detect(image_preparee)

    if visages is None:
        return []

    detections = []

    for visage in visages:
        x, y, w, h = visage[:4]
        score = float(visage[-1])

        x1 = max(0, round(x / facteur_x))
        y1 = max(0, round(y / facteur_y))
        x2 = min(largeur, round((x + w) / facteur_x))
        y2 = min(hauteur, round((y + h) / facteur_y))

        if x2 <= x1 or y2 <= y1:
            continue

        rect = (x1, y1, x2 - x1, y2 - y1)

        # Petit filtre géométrique simple
        rw, rh = rect[2], rect[3]
        ratio = rw / rh

        if rw < 40 or rh < 40:
            continue

        if ratio < 0.5 or ratio > 1.8:
            continue

        detections.append({
            "rect": rect,
            "score": score
        })

    return detections


def intersection_area(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    if xb <= xa or yb <= ya:
        return 0

    return (xb - xa) * (yb - ya)


def suppress_contained_detections(detections, containment_threshold=0.80):
    """
    Supprime les petites boîtes presque entièrement contenues
    dans une plus grande.

    Très utile ici quand YuNet détecte un œil ou la bouche
    à l'intérieur du vrai visage.
    """

    if not detections:
        return []

    detections = sorted(
        detections,
        key=lambda d: d["rect"][2] * d["rect"][3],
        reverse=True
    )

    kept = []

    for det in detections:
        rect = det["rect"]
        area = rect[2] * rect[3]

        contained = False

        for kept_det in kept:
            kept_rect = kept_det["rect"]
            inter = intersection_area(rect, kept_rect)

            if area > 0 and inter / area >= containment_threshold:
                contained = True
                break

        if not contained:
            kept.append(det)

    return kept


def select_main_face(detections, image_shape):
    """
    Choisit le visage principal dans une image censée contenir
    une seule personne.

    Critères :
    - grande taille
    - position proche du centre
    - score YuNet
    """

    if not detections:
        return None

    h, w = image_shape[:2]
    cx_img = w / 2
    cy_img = h / 2

    best_rect = None
    best_value = -1e18

    for det in detections:
        x, y, rw, rh = det["rect"]
        score = det["score"]

        area = rw * rh
        area_norm = area / (w * h)

        cx = x + rw / 2
        cy = y + rh / 2

        dx = (cx - cx_img) / (w / 2)
        dy = (cy - cy_img) / (h / 2)
        dist_centre = dx * dx + dy * dy

        # plus c'est proche du centre, mieux c'est
        centre_score = max(0.0, 1.0 - dist_centre)

        ratio = rw / rh
        ratio_score = max(0.0, 1.0 - abs(ratio - 0.8))

        valeur = (
            0.50 * area_norm +
            0.30 * centre_score +
            0.15 * score +
            0.05 * ratio_score
        )

        if valeur > best_value:
            best_value = valeur
            best_rect = det["rect"]

    return best_rect


def detect_faces(image):
    """
    Détection 'publique' :
    retourne simplement les rectangles filtrés.
    Utile pour les photos de groupe.
    """

    detections = detect_faces_detailed(image)
    detections = suppress_contained_detections(detections)

    return [d["rect"] for d in detections]


def extract_face(image, rectangle, image_size=(100, 100)):
    x, y, w, h = rectangle

    if w <= 0 or h <= 0:
        raise ValueError("Rectangle invalide")

    hauteur, largeur = image.shape[:2]

    if x < 0 or y < 0 or x + w > largeur or y + h > hauteur:
        raise ValueError("Rectangle hors image")

    visage = image[y:y+h, x:x+w]

    if visage.ndim == 3:
        visage = cv2.cvtColor(visage, cv2.COLOR_BGR2GRAY)

    visage = cv2.resize(
        visage,
        image_size,
        interpolation=cv2.INTER_AREA
    )

    return visage


def preprocess_face(image, rectangle, image_size=(100, 100)):
    visage = extract_face(image, rectangle, image_size)
    return visage.astype(np.float64).reshape(1, -1)