
# PROJ942 — Reconnaissance faciale par ACP

## Description

Système de reconnaissance faciale utilisant une Analyse
en Composantes Principales (ACP) calculée par SVD.

Détection des visages avec YuNet (OpenCV).

## Dépendances

- Python 3.11
- NumPy
- Pillow
- OpenCV 4.7.0
- Matplotlib (visualisations)

## Organisation

- acp.py : fonctions mathématiques
- face_processing.py : détection et prétraitement
- train.py : entraînement
- recognition_service.py : reconnaissance
- group_recognition.py : reconnaissance de groupe
- server.py : communication TCP avec Android

## Entraîner le modèle

python train.py

Les images d'entraînement sont classées dans
dataset/train/NomPersonne/.

## Tester une photo de groupe

Placer une photo sous :

photos_groupes/groupe.jpg

Puis lancer :

python group_recognition.py

Résultat :

resultats/groupe_annote.jpg

## Important

Le modèle doit être réentraîné si le prétraitement
des photos est modifié.

Les scores de similarité relative ne sont pas des
probabilités de reconnaissance.
