# Modélisation : Modèle Deep Learning (MLP)
# Architecture : MLP avec recherche aléatoire des hyperparamètres sur un split
# de validation interne, puis entraînement final sur le train complet.
import random
import pandas as pd
import numpy as np
import time
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from src.project_config import MODELS_DIR, PROCESSED_DIR, RESULTS_DIR, ensure_project_dirs, project_relative

tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# Espace de recherche des hyperparamètres
PARAM_SPACE = {
    "units_1":     [32, 64, 128],
    "units_2":     [16, 32, 64],
    "dropout":     [0.1, 0.2, 0.3],
    "learning_rate": [0.001, 0.005, 0.01],
    "batch_size":  [32, 64, 128],
}
N_ITER = 10  # nombre de combinaisons testées


def load_processed_data():
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze()
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def build_mlp(input_dim, units_1=32, units_2=16, dropout=0.2, learning_rate=0.001):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(units_1, activation="relu"),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(units_2, activation="relu"),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def evaluate(y_test, y_proba):
    y_pred = (y_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    return {
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "FN": cm[1, 0],
        "FP": cm[0, 1],
    }


def tune_mlp(X_fit, y_fit, X_val, y_val, class_weight_dict, input_dim):
    # Recherche aléatoire : teste N_ITER combinaisons d'hyperparamètres sur le
    # split de validation interne et retient celle qui maximise le F1.
    # RandomizedSearchCV de sklearn n'est pas compatible avec Keras, donc on
    # implémente la boucle manuellement.
    best_f1 = -1
    best_params = None

    for i in range(N_ITER):
        params = {k: random.choice(v) for k, v in PARAM_SPACE.items()}
        model = build_mlp(
            input_dim,
            units_1=params["units_1"],
            units_2=params["units_2"],
            dropout=params["dropout"],
            learning_rate=params["learning_rate"],
        )
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
        model.fit(
            X_fit, y_fit,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=params["batch_size"],
            class_weight=class_weight_dict,
            callbacks=[early_stop],
            verbose=0,
        )
        y_proba_val = model.predict(X_val, verbose=0).flatten()
        y_pred_val = (y_proba_val >= 0.5).astype(int)
        val_f1 = f1_score(y_val, y_pred_val)
        print(f"  [{i+1:02d}/{N_ITER}] {params}  ->  F1 val = {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_params = params

    print(f"\n  Meilleurs hyperparamètres : {best_params}  |  F1 val = {best_f1:.4f}")
    return best_params


def run_mlp():
    ensure_project_dirs()
    X_train, X_test, y_train, y_test = load_processed_data()

    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = {0: weights[0], 1: weights[1]}

    # Split interne pour la recherche d'hyperparamètres (ne touche pas au test set)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    print("Recherche des hyperparamètres du MLP (recherche aléatoire, 10 combinaisons)...")
    best_params = tune_mlp(X_fit, y_fit.values, X_val, y_val.values, class_weight_dict, X_train.shape[1])

    # Entraînement final sur le train complet avec les meilleurs hyperparamètres
    print("\nEntraînement final avec les meilleurs hyperparamètres...")
    model = build_mlp(
        X_train.shape[1],
        units_1=best_params["units_1"],
        units_2=best_params["units_2"],
        dropout=best_params["dropout"],
        learning_rate=best_params["learning_rate"],
    )
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=best_params["batch_size"],
        class_weight=class_weight_dict,
        callbacks=[early_stop],
        verbose=0,
    )
    train_time = time.time() - t0
    n_epochs_used = len(history.history["loss"])
    print(f"Entraînement terminé en {train_time:.2f}s ({n_epochs_used} époques effectives)")

    y_proba = model.predict(X_test, verbose=0).flatten()
    res = evaluate(y_test, y_proba)
    res.update({
        "model": "MLP (Deep Learning)",
        "train_time_s": train_time,
        "n_epochs": n_epochs_used,
        **{f"best_{k}": v for k, v in best_params.items()},
    })

    print("\n" + "=" * 80)
    print("RESULTAT DU MLP")
    print("=" * 80)
    for k, v in res.items():
        print(f"{k:25s}: {v}")

    model.save(MODELS_DIR / "mlp_model.keras")
    pd.DataFrame([res]).round(4).to_csv(RESULTS_DIR / "mlp_results.csv", index=False)
    print(f"\nModèle sauvegardé dans {project_relative(MODELS_DIR / 'mlp_model.keras')}")
    print(f"Résultats sauvegardés : {project_relative(RESULTS_DIR / 'mlp_results.csv')}")

    return model, res


if __name__ == "__main__":
    run_mlp()
