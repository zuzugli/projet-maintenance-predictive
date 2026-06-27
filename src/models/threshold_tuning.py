# Modélisation : ajustement du seuil de décision sans fuite vers le test set.

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from src.models.train_mlp import build_mlp
from src.project_config import MODELS_DIR, PROCESSED_DIR, RESULTS_DIR, ensure_project_dirs, project_relative


tf.random.set_seed(42)
np.random.seed(42)


def load_processed_data():
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze()
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def load_saved_models():
    return {
        "Régression Logistique": joblib.load(MODELS_DIR / "logreg.pkl"),
        "Random Forest": joblib.load(MODELS_DIR / "random_forest.pkl"),
        "Hist Gradient Boosting": joblib.load(MODELS_DIR / "hist_gb.pkl"),
        "MLP (Deep Learning)": keras.models.load_model(MODELS_DIR / "mlp_model.keras"),
    }


def clone_and_fit_models(X_fit, y_fit):
    """Réentraîne les modèles optimisés (mêmes hyperparamètres) sur X_fit pour choisir le seuil."""
    # Sklearn : clone avec les hyperparamètres issus de RandomizedSearchCV
    sklearn_names = {
        "Régression Logistique": "logreg.pkl",
        "Random Forest": "random_forest.pkl",
        "Hist Gradient Boosting": "hist_gb.pkl",
    }
    models = {}
    for name, fname in sklearn_names.items():
        m = clone(joblib.load(MODELS_DIR / fname))
        m.fit(X_fit, y_fit)
        models[name] = m

    # MLP : récupère les meilleurs hyperparamètres depuis mlp_results.csv
    mlp_results_path = RESULTS_DIR / "mlp_results.csv"
    if mlp_results_path.exists():
        row = pd.read_csv(mlp_results_path).iloc[0]
        mlp = build_mlp(
            input_dim=X_fit.shape[1],
            units_1=int(row["best_units_1"]),
            units_2=int(row["best_units_2"]),
            dropout=float(row["best_dropout"]),
            learning_rate=float(row["best_learning_rate"]),
        )
        batch_size = int(row["best_batch_size"])
    else:
        mlp = build_mlp(input_dim=X_fit.shape[1])
        batch_size = 64

    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_fit)
    class_weight_dict = {0: weights[0], 1: weights[1]}
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
    )
    mlp.fit(
        X_fit, y_fit,
        validation_split=0.2,
        epochs=100,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=[early_stop],
        verbose=0,
    )
    models["MLP (Deep Learning)"] = mlp

    return models


def get_proba(model, model_name, X):
    if model_name == "MLP (Deep Learning)":
        return model.predict(X, verbose=0).flatten()
    return model.predict_proba(X)[:, 1]


def evaluate_at_threshold(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "FN": cm[1, 0],
        "FP": cm[0, 1],
    }


def find_best_threshold(y_val, y_proba):
    thresholds = np.arange(0.05, 1.0, 0.05)
    scored = []
    for threshold in thresholds:
        metrics = evaluate_at_threshold(y_val, y_proba, threshold)
        metrics["threshold"] = round(float(threshold), 2)
        scored.append(metrics)
    df = pd.DataFrame(scored)
    best_row = df.loc[df["f1"].idxmax()]
    return float(best_row["threshold"]), best_row


def run_threshold_tuning():
    ensure_project_dirs()
    X_train, X_test, y_train, y_test = load_processed_data()

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=42,
    )

    print("Réentraînement des modèles optimisés sur split interne train/validation pour choisir le seuil...")
    validation_models = clone_and_fit_models(X_fit, y_fit)
    full_train_models = load_saved_models()

    rows = []
    for name, validation_model in validation_models.items():
        validation_proba = get_proba(validation_model, name, X_val)
        threshold, validation_metrics = find_best_threshold(y_val, validation_proba)

        full_model = full_train_models[name]
        test_proba = get_proba(full_model, name, X_test)
        default_metrics = evaluate_at_threshold(y_test, test_proba, 0.5)
        test_metrics = evaluate_at_threshold(y_test, test_proba, threshold)

        rows.append({
            "model": name,
            "selected_threshold": threshold,
            "validation_precision": validation_metrics["precision"],
            "validation_recall": validation_metrics["recall"],
            "validation_f1": validation_metrics["f1"],
            "validation_FN": int(validation_metrics["FN"]),
            "validation_FP": int(validation_metrics["FP"]),
            "test_f1_threshold_0_5": default_metrics["f1"],
            "test_FN_threshold_0_5": int(default_metrics["FN"]),
            "test_FP_threshold_0_5": int(default_metrics["FP"]),
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
            "test_FN": int(test_metrics["FN"]),
            "test_FP": int(test_metrics["FP"]),
        })

    df_summary = pd.DataFrame(rows).round(4)
    df_summary.to_csv(RESULTS_DIR / "threshold_tuning.csv", index=False)

    print("\n" + "=" * 120)
    print("AJUSTEMENT DU SEUIL : SEUIL CHOISI SUR VALIDATION, EVALUATION FINALE SUR TEST")
    print("=" * 120)
    print(df_summary.to_string(index=False))
    print(f"\nRésultats sauvegardés : {project_relative(RESULTS_DIR / 'threshold_tuning.csv')}")

    return df_summary


if __name__ == "__main__":
    run_threshold_tuning()
