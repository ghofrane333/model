import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Charger le modèle LightGBM
model_path = os.path.join(os.path.dirname(__file__), 'model', 'lgbm_model.pkl')
model = joblib.load(model_path)

# Charger les données d'entraînement
data_path = os.path.join(os.path.dirname(__file__), 'model', 'test_data.csv')
df = pd.read_csv(data_path)
df['SK_ID_CURR'] = df['SK_ID_CURR'].astype(int)  # S'assurer que les IDs sont des entiers

# Titre de l'application
st.title("Prédiction de crédit")

# Fonction pour la prédiction d'un client spécifique
def predict_client(ID):
    seuil = 0.625

    # Vérifier si l'ID existe dans la base de données
    if df[df['SK_ID_CURR'] == ID].empty:
        st.error("Ce client n'est pas répertorié")
        return

    # Extraire les données du client
    X = df[df['SK_ID_CURR'] == ID].drop(['SK_ID_CURR'], axis=1)

    # Vérifier et réorganiser les colonnes pour correspondre au modèle
    expected_columns = model.feature_name_
    missing_cols = set(expected_columns) - set(X.columns)
    for col in missing_cols:
        X[col] = 0
    X = X[expected_columns]

    if X.shape[1] != model.n_features_in_:
        st.error(f"Nombre de caractéristiques incorrect: attendu {model.n_features_in_}, reçu {X.shape[1]}")
        return

    # Prédiction
    try:
        probability_default_payment = model.predict_proba(X)[:, 1][0]
    except Exception as e:
        st.error(f"Erreur lors de la prédiction: {str(e)}")
        return

    prediction = "Prêt NON Accordé, risque de défaut" if probability_default_payment >= seuil else "Prêt Accordé"
    st.success(f"Probabilité de défaut de paiement: {probability_default_payment:.4f}")
    st.write(f"Prédiction: {prediction}")

# Interface utilisateur Streamlit
st.sidebar.header("Entrez l'ID client")
client_id = st.sidebar.text_input("ID client", "")

if client_id:
    try:
        client_id = int(client_id)
        predict_client(client_id)
    except ValueError:
        st.error("L'ID client doit être un nombre entier")

# Fonction pour prédire pour tous les clients
def predict_all_clients():
    try:
        Xtot = df.drop(['SK_ID_CURR'], axis=1)
        seuil = 0.625
        y_pred = model.predict_proba(Xtot)[:, 1]
        y_seuil = y_pred >= seuil
        y_seuil = np.array(y_seuil > 0) * 1

        df_pred = df.copy()
        df_pred['Proba'] = y_pred
        df_pred['PREDICTION'] = y_seuil

        st.write(df_pred[['SK_ID_CURR', 'Proba', 'PREDICTION']].head())
        st.download_button(
            label="Télécharger les prédictions",
            data=df_pred.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Erreur lors de la génération des prédictions: {str(e)}")

# Afficher les prédictions pour tous les clients
if st.button("Afficher les prédictions pour tous les clients"):
    predict_all_clients()



