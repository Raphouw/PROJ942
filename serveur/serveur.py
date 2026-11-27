import os
import time
import random
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))
DOSSIER = os.path.join(BASE, "photos")
DATASET = os.path.join(BASE, "Smile")
os.makedirs(DOSSIER, exist_ok=True)


def sauvegarder(donnees):
    nom = "photo_" + str(int(time.time() * 1000)) + ".jpg"
    chemin = os.path.join(DOSSIER, nom)
    with open(chemin, "wb") as f:
        f.write(donnees)
    return nom, chemin


# ---- Module de reconnaissance ----
# FACTICE pour l'instant : sera remplace par la fonction de l'ACP
def reconnaitre(chemin_image):
    personnes = sorted(os.listdir(DATASET))
    return random.choice(personnes), round(random.uniform(0.5, 0.99), 2)


# Envoyer une photo : POST /photos
@app.route("/photos", methods=["POST"])
def recevoir_photo():
    donnees = request.get_data()
    if len(donnees) == 0:
        return jsonify({"erreur": "aucune donnee"}), 400
    nom, _ = sauvegarder(donnees)
    print("Photo recue :", nom, len(donnees), "octets")
    return jsonify({"nom": nom}), 200


# Envoyer une photo ET obtenir la reconnaissance : POST /reconnaitre
@app.route("/reconnaitre", methods=["POST"])
def route_reconnaitre():
    donnees = request.get_data()
    if len(donnees) == 0:
        return jsonify({"erreur": "aucune donnee"}), 400
    nom, chemin = sauvegarder(donnees)
    personne, score = reconnaitre(chemin)
    print("Reconnaissance :", nom, "->", personne, score)
    return jsonify({"fichier": nom, "nom": personne, "score": score}), 200


# Liste des photos : GET /photos
@app.route("/photos", methods=["GET"])
def liste_photos():
    noms = sorted(os.listdir(DOSSIER), reverse=True)
    return jsonify(noms)


# Une photo : GET /photos/<nom>
@app.route("/photos/<nom>", methods=["GET"])
def une_photo(nom):
    return send_from_directory(DOSSIER, nom)


# Page web pour voir toutes les photos : GET /voir
@app.route("/voir")
def voir():
    noms = sorted(os.listdir(DOSSIER), reverse=True)
    html = "<h1>Photos recues</h1>"
    for n in noms:
        html += '<img src="/photos/' + n + '" width="300"> '
    return html


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
