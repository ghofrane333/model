from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Charger le modèle LightGBM
model_path = os.path.join(os.path.dirname(__file__), 'model', 'lgbm_model.pkl')
model = joblib.load(model_path)

# Charger les données d'entraînement
data_path = os.path.join(os.path.dirname(__file__), 'model', 'test_data.csv')
df = pd.read_csv(data_path)
df['SK_ID_CURR'] = df['SK_ID_CURR'].astype(int)  # S'assurer que les IDs sont des entiers

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    all_id_client = list(df['SK_ID_CURR'].unique())
    seuil = 0.625

    # Récupérer l'ID client du formulaire
    data = request.form
    ID = data.get('id_client', '')

    try:
        ID = int(ID)  # Convertir l'ID en entier
    except ValueError:
        return jsonify({"error": "ID invalide"}), 400

    # Vérifier si l'ID existe dans la base de données
    if df[df['SK_ID_CURR'] == ID].empty:
        return jsonify({"error": "Ce client n'est pas répertorié"}), 404

    # Extraire les données du client
    X = df[df['SK_ID_CURR'] == ID].drop(['SK_ID_CURR'], axis=1)

    # Vérifier et réorganiser les colonnes pour correspondre au modèle
    expected_columns = model.feature_name_
    missing_cols = set(expected_columns) - set(X.columns)
    for col in missing_cols:
        X[col] = 0
    X = X[expected_columns]

    if X.shape[1] != model.n_features_in_:
        return jsonify({
            "error": "Nombre de caractéristiques incorrect",
            "expected_features_count": model.n_features_in_,
            "received_features_count": X.shape[1]
        }), 400

    # Prédiction
    try:
        probability_default_payment = model.predict_proba(X)[:, 1][0]
    except Exception as e:
        return jsonify({'error': f'Erreur lors de la prédiction: {str(e)}'}), 500

    prediction = "Prêt NON Accordé, risque de défaut" if probability_default_payment >= seuil else "Prêt Accordé"
    return jsonify({"probability": probability_default_payment, "prediction": prediction})

@app.route('/prediction_complete')
def pred_model():
    try:
        Xtot = df.drop(['SK_ID_CURR'], axis=1)
        seuil = 0.625
        y_pred = model.predict_proba(Xtot)[:, 1]
        y_seuil = y_pred >= seuil
        y_seuil = np.array(y_seuil > 0) * 1

        df_pred = df.copy()
        df_pred['Proba'] = y_pred
        df_pred['PREDICTION'] = y_seuil

        return df_pred.to_json(orient='index')

    except Exception as e:
        return jsonify({'error': f'Erreur lors de la génération des prédictions: {str(e)}'}), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))  # Changer ici pour 8000
    app.run(host='0.0.0.0', port=port)


