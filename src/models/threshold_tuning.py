# Modélisation : Ajustement du seuil de décision

# Objectif : par défaut, un modèle de classification utilise un seuil
# de 0.5. Sur un dataset déséquilibré, ce seuil n'est pas forcément optimal. 
# On teste plusieurs seuils sur chaque modèle pour que la comparaison
# finale se fasse entre modèles à leur meilleur seuil respectif, pas
# Pour chaque modèle, on identifie le seuil qui maximise le F1.

import pandas as pd
import numpy as np
import joblib
from tensorflow import keras
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

PROCESSED_DIR = "data/processed"
MODELS_DIR = "outputs/models"


def load_processed_data():
    # Recharge les données déjà préparées par le pipeline de preprocessing
    X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
    y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv").squeeze()
    return X_test, y_test


def load_saved_models():
    # Charge les modèles déjà entraînés et sauvegardés - pas besoin de réentraîner
    return {
        "Régression Logistique": joblib.load(f"{MODELS_DIR}/logreg.pkl"),
        "Random Forest": joblib.load(f"{MODELS_DIR}/random_forest.pkl"),
        "Hist Gradient Boosting": joblib.load(f"{MODELS_DIR}/hist_gb.pkl"),
        "MLP (Deep Learning)": keras.models.load_model(f"{MODELS_DIR}/mlp_model.keras"),
    }


def evaluate_at_threshold(y_test, y_proba, threshold):
    # Applique un seuil donné aux probabilités prédites et calcule les métriques à ce seuil
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    return {
        "threshold": round(threshold, 2),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "FN": cm[1, 0],
        "FP": cm[0, 1],
    }


def find_best_threshold(y_test, y_proba):
    # Teste une grille de seuils (0.05 à 0.95) et renvoie celui qui maximise le F1
    thresholds = np.arange(0.05, 1.0, 0.05)
    results = [evaluate_at_threshold(y_test, y_proba, t) for t in thresholds]
    df = pd.DataFrame(results)
    best_row = df.loc[df["f1"].idxmax()]
    return best_row, df


def run_threshold_tuning():
    # Pour chacun des 4 modèles : trouve le seuil qui maximise le F1, et compare
    # ce résultat au seuil par défaut (0.5), pour mesurer le gain de l'ajustement
    X_test, y_test = load_processed_data()
    models = load_saved_models()

    summary = []

    for name, model in models.items():
        if name == "MLP (Deep Learning)":
            y_proba = model.predict(X_test, verbose=0).flatten()
        else:
            y_proba = model.predict_proba(X_test)[:, 1]

        default_res = evaluate_at_threshold(y_test, y_proba, 0.5)
        best_res, _ = find_best_threshold(y_test, y_proba)
        summary.append({
            "model": name,
            "f1_seuil_0.5": default_res["f1"], "FN_seuil_0.5": default_res["FN"], "FP_seuil_0.5": default_res["FP"],
            "meilleur_seuil": best_res["threshold"],
            "f1_meilleur_seuil": best_res["f1"], "FN_meilleur_seuil": best_res["FN"], "FP_meilleur_seuil": best_res["FP"],
        })

    df_summary = pd.DataFrame(summary).round(4)
    print("\n" + "=" * 110)
    print("AJUSTEMENT DU SEUIL : COMPARAISON SUR LES 4 MODELES")
    print("=" * 110)
    print(df_summary.to_string(index=False))

    return df_summary


if __name__ == "__main__":
    run_threshold_tuning()
    