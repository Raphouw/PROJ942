
import cv2
import numpy as np

from acp import load_model, project, predict, compare_all

from face_processing import (
    detect_faces_detailed,
    suppress_contained_detections,
    select_main_face,
    preprocess_face
)


class RecognitionService:

    def __init__(self, model_path, seuil=None):
        """
        Charge le modèle ACP une seule fois.
        """
        self.model = load_model(model_path)
        self.seuil = seuil

    def preprocess_image(self, image_bytes):
        """
        Prépare une photo exactement comme
        les photos utilisées à l'entraînement.
        """

        # 1. Décoder le JPEG
        tableau = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(tableau, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Impossible de décoder l'image")

        # 2. Détecter les visages
        detections = detect_faces_detailed(image)

        # 3. Supprimer les petites détections incluses
        detections = suppress_contained_detections(detections)

        if len(detections) == 0:
            raise ValueError("Aucun visage détecté")

        # 4. Sélectionner le visage principal
        rectangle = select_main_face(detections, image.shape)

        if rectangle is None:
            raise ValueError("Impossible de sélectionner un visage")

        # 5. Extraire et préparer le visage
        X_test = preprocess_face(
            image,
            rectangle,
            self.model["image_size"]
        )

        return X_test

    def analyser_image(self, image_bytes):
        """
        Identifie le visage principal d'une photo.
        """

        # 1. Prétraitement
        X_test = self.preprocess_image(image_bytes)

        # 2. Projection dans l'espace ACP
        Y_test = project(
            self.model["W"],
            X_test,
            self.model["mu"]
        )

        # 3. Prédiction de l'identité
        resultat = predict(
            Y_test,
            self.model["Y_train"],
            self.model["y_train"],
            self.model["labels"],
            seuil=self.seuil
        )

        # 4. Comparaison avec toutes les personnes
        comparaisons = compare_all(
            Y_test,
            self.model["Y_train"],
            self.model["y_train"],
            self.model["labels"]
        )

        resultat["comparaisons"] = comparaisons

        return resultat
