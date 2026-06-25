# Modélisation : Baseline (régression logistique sans traitement du déséquilibre)

# objectif : avoir un premier point de référence chiffré, et montrer concrètement
# pourquoi l'accuracy seule est insuffisante sur ce dataset déséquilibré

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, classification_report
)

PROCESSED_DIR = "data/processed"


def load_processed_data():
    # Recharge les données déjà préparées par le pipeline de preprocessing
    X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv")
    X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
    y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train.csv").squeeze()
    y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test, model_name="Modèle"):
    # Calcule toutes les métriques pertinentes pour un dataset déséquilibré :
    # accuracy, recall, f1, roc-auc et pr-auc
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*60}\n{model_name}\n{'='*60}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")
    print(f"\nMatrice de confusion :\n{cm}")
    print(f"  -> Vrais négatifs (TN)  : {cm[0,0]}")
    print(f"  -> Faux positifs (FP)   : {cm[0,1]}")
    print(f"  -> Faux négatifs (FN)   : {cm[1,0]}  <- pannes manquées")
    print(f"  -> Vrais positifs (TP)  : {cm[1,1]}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Pas de panne', 'Panne'])}")

    return {"model": model_name, "accuracy": acc, "recall": recall,
            "f1": f1, "roc_auc": roc_auc, "pr_auc": pr_auc}


def run_baseline():
    # Entraîne la régression logistique baseline et l'évalue
    X_train, X_test, y_train, y_test = load_processed_data()

    # On compare aussi à un "modèle naïf" qui prédit toujours la classe majoritaire
    # pour matérialiser le piège de l'accuracy
    naive_accuracy = (y_test == 0).mean()
    print(f"Un modèle qui prédirait toujours 'pas de panne' aurait une accuracy de {naive_accuracy:.4f}, "
          f"tout en ratant 100% des vraies pannes (Recall = 0).")

    # max_iter augmenté car la régression logistique peut avoir besoin de plus
    # d'itérations pour converger sur ce nombre de features (14 colonnes après encodage)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    results = evaluate_model(model, X_test, y_test, model_name="Baseline - Régression Logistique (sans rééquilibrage)")
    return model, results


if __name__ == "__main__":
    run_baseline()