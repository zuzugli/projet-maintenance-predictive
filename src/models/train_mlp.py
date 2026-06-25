# Modélisation : Modèle Deep Learning (MLP)
# Architecture : MLP simple (2 couches cachées), avec class_weight pour gérer le déséquilibre
import pandas as pd
import numpy as np
import time
import os
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
PROCESSED_DIR = "data/processed"
MODELS_DIR = "outputs/models"
# Seed fixe pour la reproductibilité
tf.random.set_seed(42)
np.random.seed(42)
def load_processed_data():
    # Recharge les données déjà préparées par le pipeline de preprocessing
    X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv")
    X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
    y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train.csv").squeeze()
    y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test
def build_mlp(input_dim):
    # Architecture simple :
    # - 2 couches cachées (32 puis 16 neurones), activation ReLU
    # - Dropout après chaque couche cachée : désactive aléatoirement une fraction
    # des neurones pendant l'entraînement, pour limiter le surapprentissage
    # - 1 neurone de sortie, activation sigmoid : produit une probabilité entre
    # 0 et 1 (classification binaire), comme predict_proba pour les modèles sklearn.
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
def evaluate(y_test, y_proba):
    # Calcule les métriques importantes
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
def run_mlp():
    # Entraîne le MLP et l'évalue, avec gestion du déséquilibre par class_weight
    X_train, X_test, y_train, y_test = load_processed_data()
    # class_weight calculé de la même façon que pour les modèles sklearn,
    # mais Keras attend un dictionnaire {classe: poids} plutôt qu'une chaîne "balanced"
    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = {0: weights[0], 1: weights[1]}
    print(f"Poids de classes appliqués : {class_weight_dict}")
    model = build_mlp(input_dim=X_train.shape[1])
    # EarlyStopping : arrête l'entraînement si la performance sur la validation
    # ne s'améliore plus pendant 10 époques, et restaure les meilleurs poids
    # Sécurité supplémentaire contre le surapprentissage.
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=64,
        class_weight=class_weight_dict,
        callbacks=[early_stop],
        verbose=0,
    )
    train_time = time.time() - t0
    n_epochs_used = len(history.history["loss"])
    print(f"Entraînement terminé en {train_time:.2f}s ({n_epochs_used} époques effectives sur 100 max, "
          f"arrêt anticipé si la validation ne progresse plus)")
    y_proba = model.predict(X_test, verbose=0).flatten()
    res = evaluate(y_test, y_proba)
    res.update({"model": "MLP (Deep Learning)", "train_time_s": train_time, "n_epochs": n_epochs_used})
    print("\n" + "=" * 80)
    print("RESULTAT DU MLP")
    print("=" * 80)
    for k, v in res.items():
        print(f"{k:15s}: {v}")

    # Sauvegarde du modèle entraîné (format Keras natif .keras), pour être
    # réutilisé en D6 (synthèse finale) sans avoir à le réentraîner depuis zéro.
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(f"{MODELS_DIR}/mlp_model.keras")
    print(f"\nModèle sauvegardé dans {MODELS_DIR}/mlp_model.keras")

    return model, res
if __name__ == "__main__":
    run_mlp()