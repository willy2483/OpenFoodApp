import os
import wget
import pandas as pd
import numpy as np
from joblib import dump, load
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

################# lecture du csv #######################

def csv_reader(row_number, csv_path, csv_url):
    """
    Lecture du csv
    """
    if not os.path.exists(csv_path):
        print('downloading ', csv_url)
        wget.download(csv_url,csv_path)
        print('downloaded', csv_path)
    print("reading " + csv_path + " to DataFrame")
    df = pd.read_csv(
    csv_path,
    sep='\t', 
    low_memory =False ,
    encoding='utf-8', 
    nrows = row_number)
    print("csv to DataFrame finished!")
    return df


################ suppression des colonnes/lignes #####################

def raw_cleanse(df):
    """
    Nettoyage de la raw_data
    """

    df_100g = df.filter(regex=("_100g"))
    # suite à l'analyse des données parmi les variables autres 
    # que les variables quantitatives on garde seulement le nutriscore_grade


    # suite à l'analyse des corrélations on enlève la colonne 'nutrition-score-fr_100g' 
    # qui n'est pas disponible pour l'utilisateur final et qui est trop corrélée à la note,
    # on supprime la colonne 'soduim_100g' qui est très corrélée avec le sel et qui n'est pas
    # directement disponible pour l'utilisateur final
    df_100g = df_100g.drop(['nutrition-score-fr_100g','sodium_100g'], axis = 1)

    df = df[['nutriscore_grade']].join(df_100g)

    # on supprime tous les enregistrements qui n'ont pas de variable cible
    df.dropna(subset = ['nutriscore_grade', ], inplace = True)

    # on supprime les doublons
    df.drop_duplicates(keep="last",inplace=True)

    # on supprime toutes les colonnes vides
    df.dropna(axis=1, how='all',inplace = True)

    # on ne garde que les colonnes suffisament remplies
    seuil_remplissage = 0.25
    useful_col = [col for col in df.columns if df[col].isna().mean() < seuil_remplissage]

    clean_df = df[useful_col]

    return(clean_df)

############################# encodage #############################

def encoding_grade(clean_df):

    """
    Encoding de la variable qualitative cible
    """

# on réalise l'encoding sans appeler les modules de preprocess
# on pourra le changer après
    encod_df = clean_df[clean_df.columns[1:]]

    encoding = {
        'a':5,
        'b':4,  
        'c':3,
        'd':2,
        'e':1,
    }

    encod_df.insert(0,'nutriscore_grade',clean_df['nutriscore_grade'].map(encoding))

    return(encod_df)


######################## traitement des valeurs aberrantes ##############################

def outlier_values(datacolumn):
    """
    Donne les bornes au-delà desquelles une valeur est considérée comme aberrante,
    selon le critère IRQ
    """
    Q1, Q3 = datacolumn.quantile([0.25,0.75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    return lower_range,upper_range

def outlier_cleanse(encod_df,clean_df):

    """
    Suppression des valeurs aberrantes dans le dataframe encodé et le dataframe final
    """

    for col in encod_df.columns:
        low, up = outlier_values(encod_df[col])
        
        # la liste des index sera utiliser pour dropper sur clean_df
        index_list = encod_df[(encod_df[col] > up) | (encod_df[col] < low)].index
        encod_df = encod_df.drop(index_list)
        clean_df = clean_df.drop(index_list)
    
    return(encod_df,clean_df)

##################### suppression des variables explicatives inutiles #####################

def corr_cleanse(encod_df, clean_df):

    """
    Nettoyage des colonnes selon le critère des corrélations
    """

    # création de la matrice de corrélation
    df_corr = encod_df.corr(method = 'pearson')

    # on élimine les variables explicatives qui ne sont pas suffisament corrélées avec la variable cible
    seuil_sign = 0.05
    col_list = [col for col in encod_df.columns if np.abs(df_corr['nutriscore_grade'][col]) < seuil_sign]
    clean_df = clean_df.drop(col_list, axis = 1)

    return(clean_df)

# on fait le choix de supprimer le nutriscore qui est une variable qui détermine trop le grade et 
# qui ne sera pas disponible pour l'utilisateur, elle risque d'influer sur le résultat du modèle 
# on garde energie kcal très corrélé à energie100g (mieux remplie que energie kcal) et sodium qui 
# sont très corrélé au sel car doivent être disponible pour l'utilisateur final leur forte corrélation 
# n'influe pas sur le résultat du modèle


def data_cleaning(nb_lines, chemin_csv, chemin_url):

    df = csv_reader(nb_lines,chemin_csv, chemin_url)
    clean_df = raw_cleanse(df)
    encod_df = encoding_grade(clean_df)
    encod_df, clean_df = outlier_cleanse(encod_df,clean_df)
    clean_df = corr_cleanse(encod_df, clean_df)

    return clean_df


########################### équilibrage de l'échantillon ################################

def down_sampling(clean_df):

    # taille de la classe moins représentée
    grade_list=['a','b','c','d','e']
    min_size = min([len(clean_df[clean_df['nutriscore_grade'] == grade]) for grade in grade_list])

    # équilibrage des classes
    frames = [clean_df[clean_df['nutriscore_grade'] == grade].sample(min_size) for grade in grade_list]

    # nouveau dataframe
    down_samp = pd.concat(frames)

    return(down_samp)


##################### séparation des jeux de données ###########################

def split_sample(sample_df, target_var):

    """
    Création du jeu d'entraînement et de test à partir de l'échantillon équilibré
    """
    
    # on rajoute le stratify pour être sûr de la répartition équitable de chaque classe
    samp_X = sample_df.drop(target_var,axis=1)
    samp_y = sample_df[target_var]


    X_train, X_test, y_train, y_test = train_test_split(

        # pas besoin d'encoder la variable, mais attention à ne pas inclure la variable cible dans
        # le dataframe de X
        samp_X, 
        samp_y, 
        test_size=0.3, 
        random_state=42, 
        stratify=samp_y)

    return(X_train, X_test, y_train, y_test)


################# lecture du csv #######################
nb_lines = 1000000
chemin_csv = "en.openfoodfacts.org.products.csv"
chemin_url = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv"
tar_var = 'nutriscore_grade'
fill_strat = 'median'
model_param = 13

# création du jeu de données
X_train, X_test, y_train, y_test = split_sample(down_sampling(data_cleaning(nb_lines, chemin_csv, chemin_url)), tar_var)

# création du pipeline
nutri_pipe = Pipeline([

    ##################### fonctions de preprocessing du pipeline ##################
    ('filler', SimpleImputer(missing_values=np.nan, strategy=fill_strat)), 
    ('standardizer', MinMaxScaler()),

    ####################### spécification du modèle #################################

    ('rfc', RandomForestClassifier(max_depth= model_param, random_state=0))])

# Entraînement du modèle
nutri_pipe.fit(X_train,y_train)

####################### export joblib ####################################

dump(nutri_pipe,'pipeline_export.joblib')
