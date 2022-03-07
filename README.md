# Projet 1 - Groupe 4

## Utilisation du module pipeline.py

Le module pipeline.py permet de générer un export .joblib d'un modèle entraîné de Machine Learning,
les données en provenance de Openfoodfacts sont téléchargées, nettoyées et entraîne notre modèle.

A l'issue de l'entraînement, l'export pipeline_export.joblib est crée. S'il n'a jamais était créé:

* Lancer la commande suivante à la racine du répertoire "projet-1-groupe4":

```bash
python3 pipeline.py
```

* le fichier "pipeline_export.joblib" sera crée à la racine du répertoire "projet-1-groupe4"

## Utilisation de l'export "pipeline_expor.joblib"

* Créer un environnement virtuel dans votre répertoire, et installer le package joblib, avec pipenv:

```bash
pipenv install pandas
pipenv install joblib
pipenv install sklearn
```

* Dans un module Python, entrer les lignes suivantes:

```python
import numpy as np
import pandas as pd 
from joblib import load
from sklearn.pipeline import Pipeline

nutri_pipe = load('./chemin_vers_le_pipeline/pipeline_export.joblib')
```

Bravo! vous avez chargé l'export ! Vous pouvez faire de nouvelles prédictions:

* Crée un nouveau DataFrame sur le schéma suivant:

```python
new_predict = pd.DataFrame([
    [141, 592, 5.9, 1.5, 13, 0.7, 8.6, 0.67], # poisson à l'andalouse nutriscore A
    [nouvelle_liste_de_valeurs],
    [une_autre_liste_de_valeurs],
])
```

* Les valeurs à renseigner sont celles figurant dans le résultat de la commande suivante:

```python
print(nutri_pipe.named_steps['filler'].feature_names_in_)
```
* Si vous souhaitez rentrer les valeurs d'un aliment disposant d'une fiche nutriscore, les valeurs à rentrer dans chacune des listes du DataFrame correspondent à:  

```python
[  
    Energie pour 100g (kJ),  
    Energie pour 100g (kcal),  
    Total de Matières grasses pour 100g (g),  
    Acides gras saturés pour 100g (g) (inclus dans les matières grasses),  
    Total des glucides pour 100g (g),  
    Sucres pour 100g (g) (inclus dans les glucides),  
    Protéines pour 100g (g),  
    Sel pour 100g (g)  
]
```

Remarques:
* Les fibres alimentaires ne sont pas nécessaires dans notre modèle
* Si vous n'avez pas une ou plusieurs de ces informations, renseignez-les comme np.nan, si vous voulez éviter le message d'avertissement qui vous indique que ces valeurs sont considérées comme correspondantes aux features du modèle, explicitez les colonnes de votre DataFrame

```python
new_predict.columns = nutri_pipe.named_steps['filler'].feature_names_in_
```
Voilà! Vous pouvez maintenant faire une nouvelle prédiction!

```python
y_pred = nutri_pipe.predict(new_predict)
print(y_pred)

prob_mat= nutri_pipe.predict_proba(new_predict)
print(prob_mat)
```
