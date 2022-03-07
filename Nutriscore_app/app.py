# importer la class `Flask`
from joblib import load
from sklearn.pipeline import Pipeline
from flask import Flask, render_template, request
import numpy as np
import pandas as pd

# instancier une application Flask
app = Flask(__name__)

# définir la route `home`
# il faut rajouter POST pour envoyer le résutat du form
@app.route("/", methods=["GET", "POST"])

def home():

    # pour l'instant le render template ne fait que renvoyer une string
    # on peut rajouter des variables pour rendre le dynamique en utilisant vraiment
    # le moteur de rendu Jinja
    
    
    if request.method == "POST":
        energy_100g = request.form.get("energy-100g")
        energy_kcal_100g = request.form.get("energy-kcal_100g")
        fat_100g = request.form.get("fat_100g")
        saturated_fat_100g = request.form.get("saturated-fat_100g")
        carbohydrates_100g = request.form.get("carbohydrates_100g")
        sugar_100g = request.form.get("sugar_100g")
        proteins_100g = request.form.get("proteins_100g")
        salt_100g = request.form.get("salt_100g")

        list_get = [energy_100g, energy_kcal_100g, fat_100g, saturated_fat_100g, carbohydrates_100g, sugar_100g, proteins_100g, salt_100g]

        
        # si la list ne contient que des chaînes vides
        if all(elem == '' for elem in list_get):
            
            return render_template('home.html', void_get = True)
        
        
        # si on a au moins une valeur rentrée par l'utilisateur
        else:
            pipe_export = load('./pipeline_export.joblib')
            

            # list_pred contient la liste des valeurs renseignées comme premier élément et l'information s'il manque des valeurs
            # lors de l'envoi du formulaire comme deuxième élément
            list_pred = [[np.nan if elt =='' else elt for elt in list_get], True if '' in list_get else False]      
            X_pred = pd.DataFrame([list_pred[0]])
            X_pred.columns = pipe_export.named_steps['filler'].feature_names_in_
            y_pred = pipe_export.predict(X_pred)
            
            
            return render_template('home.html', y_pred = y_pred, inc_list = list_pred[1])
            #, heure_actuelle = now, prenom= prenom, nom= nom)

    elif request.method == "GET":
        return render_template('home.html')
