import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
from pytorch_tabnet.tab_model import TabNetRegressor

<<<<<<< HEAD
=======


>>>>>>> 0a46e7f (Déploiement app TabNet Streamlit)
# Chargement

# Charger le CSV pour obtenir les bornes des années
df_years = pd.read_csv("investments_VC.csv", encoding='unicode_escape')
df_years.columns = df_years.columns.str.strip().str.lower()
annee_min = int(df_years['founded_year'].min())
annee_max = int(df_years['founded_year'].max())

columns = joblib.load("columns.pkl")
label_encoders = joblib.load("label_encoders.pkl")

model = TabNetRegressor()
model.load_model("tabnet_model.zip")

st.title("Prédiction de levée de fonds avec TabNet")

# Formulaire

form = {}
for col in columns:
    if col in label_encoders:
        options = label_encoders[col].classes_
        form[col] = st.selectbox(col, options)
    elif col == "funding_rounds":
        form[col] = st.selectbox(col, options=[1, 2, 3])
    elif col == "founded_year":
        form[col] = st.selectbox(col, options=list(range(annee_min, annee_max + 1)))
    else:
        form[col] = st.number_input(col, step=1)

if st.button("Prédire"):
    df_input = pd.DataFrame([form])

    # Vérification des valeurs numériques raisonnables
    founded_year = int(df_input["founded_year"].values[0])
    funding_rounds = int(df_input["funding_rounds"].values[0])
    
    if not (1970 <= founded_year <= 2025):
        st.error("Veuillez entrer une année de création réaliste (entre 1970 et 2025).")
    elif not (1 <= funding_rounds <= 50):
        st.error("Veuillez entrer un nombre de tours de financement raisonnable (entre 1 et 50).")
    else:
        # Encodage des colonnes catégorielles
        for col in label_encoders:
            df_input[col] = label_encoders[col].transform(df_input[col].astype(str))
        
        # Prédiction
        prediction = model.predict(df_input.values)
        log_pred = prediction[0][0]

        # Protection contre valeurs aberrantes
        if log_pred > 30 or log_pred < 0:
            st.error(f"Prédiction log trop extrême : {log_pred:.2f}. Vérifiez les valeurs saisies.")
        else:
            montant = np.expm1(log_pred)
            montant_reel = f"{montant:,.0f} $".replace(",", " ")
<<<<<<< HEAD
            st.success(f"Montant estimé levé : {montant_reel} (log = {log_pred:.2f})")
=======
            st.success(f"Montant estimé levé : {montant_reel} (log = {log_pred:.2f})")
>>>>>>> 0a46e7f (Déploiement app TabNet Streamlit)
