# Modélisation : Ajustement du seuil de décision

# Objectif : par défaut, un modèle de classification utilise un seuil
# de 0.5. Sur un dataset déséquilibré, ce seuil n'est pas forcément optimal. 
# On teste plusieurs seuils sur chaque modèle pour que la comparaison
# finale se fasse entre modèles à leur meilleur seuil respectif, pas
# Pour chaque modèle, on identifie le seuil qui maximise le F1.

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

PROCESSED_DIR = "data/processed"

tf.random.set_seed(42)
np.random.seed(42)


def load_processed_data():
    # Recharge les données déjà préparées par le pipeline de preprocessing
    X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv")
    X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
    y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train.csv").squeeze()
    y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def get_trained_models(X_train, y_train):
    # Reentraine les 3 modèles sklearn et les renvoie dans un dictionnaire {nom: modèle}
    models = {}

    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr.fit(X_train, y_train)
    models["Régression Logistique"] = lr

    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42,
                                  class_weight="balanced", n_jobs=-1)
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf

    gb = HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=0.1,
                                          random_state=42, class_weight="balanced")
    gb.fit(X_train, y_train)
    models["Hist Gradient Boosting"] = gb

    return models


def build_and_train_mlp(X_train, y_train):
    # Réentraîne le MLP avec class_weight pour gérer le déséquilibre
    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = {0: weights[0], 1: weights[1]}

    model = keras.Sequential([
        keras.layers.Input(shape=(X_train.shape[1],)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=64,
              class_weight=class_weight_dict, callbacks=[early_stop], verbose=0)
    return model


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
    X_train, X_test, y_train, y_test = load_processed_data()

    sklearn_models = get_trained_models(X_train, y_train)
    mlp_model = build_and_train_mlp(X_train, y_train)

    summary = []

    # Modèles sklearn (LogReg, RF, Hist GB)
    for name, model in sklearn_models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        default_res = evaluate_at_threshold(y_test, y_proba, 0.5)
        best_res, _ = find_best_threshold(y_test, y_proba)
        summary.append({
            "model": name,
            "f1_seuil_0.5": default_res["f1"], "FN_seuil_0.5": default_res["FN"], "FP_seuil_0.5": default_res["FP"],
            "meilleur_seuil": best_res["threshold"],
            "f1_meilleur_seuil": best_res["f1"], "FN_meilleur_seuil": best_res["FN"], "FP_meilleur_seuil": best_res["FP"],
        })

    # MLP (probabilités obtenues différemment : .predict() au lieu de .predict_proba())
    y_proba_mlp = mlp_model.predict(X_test, verbose=0).flatten()
    default_res = evaluate_at_threshold(y_test, y_proba_mlp, 0.5)
    best_res, _ = find_best_threshold(y_test, y_proba_mlp)
    summary.append({
        "model": "MLP (Deep Learning)",
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