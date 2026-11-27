import cv2

import config

from face_processing import (
    detect_faces_detailed,
    suppress_contained_detections,
    select_main_face
)

dataset = config.TRAIN
sortie = config.RESULTATS / "debug_detection"
sortie.mkdir(parents=True, exist_ok=True)

for dossier in sorted(dataset.iterdir()):
    if not dossier.is_dir():
        continue

    destination = sortie / dossier.name
    destination.mkdir(parents=True, exist_ok=True)

    for fichier in sorted(dossier.iterdir()):
        if fichier.suffix.lower() not in (".jpg", ".jpeg", ".png", ".pgm"):
            continue

        image = cv2.imread(str(fichier))

        if image is None:
            print(f"Image illisible : {fichier}")
            continue

        detections = detect_faces_detailed(image)
        detections = suppress_contained_detections(detections)
        rectangle_principal = select_main_face(detections, image.shape)

        annotee = image.copy()

        # Toutes les détections en vert
        for det in detections:
            x, y, w, h = det["rect"]
            cv2.rectangle(
                annotee,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

        # Visage principal en rouge
        if rectangle_principal is not None:
            x, y, w, h = rectangle_principal
            cv2.rectangle(
                annotee,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                3
            )

        cv2.imwrite(
            str(destination / fichier.name),
            annotee
        )

        print(
            f"{dossier.name}/{fichier.name} : "
            f"{len(detections)} détection(s)"
        )

print("\nImages de diagnostic enregistrées.")