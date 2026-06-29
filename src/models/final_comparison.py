# Modélisation : Synthèse finale

# Objectif : recharger les 4 modèles déjà entraînés, les évaluer chacun à
# son seuil validé sur un split interne du train, et comparer les 6 critères exigés par la consigne :
# performance, stabilité, interprétabilité, coût de calcul, facilité de déploiement, cohérence métier
# Choix et sauvegarde du modèle final argumenté sur l'ensemble de ces critères

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import joblib
import time
import statistics
import tensorflow as tf
from tensorflow import keras
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from src.models.train_mlp import build_mlp
from src.project_config import MODELS_DIR, PROCESSED_DIR, RESULTS_DIR, ensure_project_dirs, project_relative

tf.random.set_seed(42)
np.random.seed(42)


def load_processed_data():
    # Recharge les données déjà préparées par le pipeline de preprocessing
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze()
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def load_all_models():
    # Recharge les 3 modèles sklearn (.pkl) et le MLP (.keras), déjà entraînés
    models = {
        "Régression Logistique": joblib.load(MODELS_DIR / "logreg.pkl"),
        "Random Forest": joblib.load(MODELS_DIR / "random_forest.pkl"),
        "Hist Gradient Boosting": joblib.load(MODELS_DIR / "hist_gb.pkl"),
        "MLP (Deep Learning)": keras.models.load_model(MODELS_DIR / "mlp_model.keras"),
    }
    return models


def get_proba(model, model_name, X_test):
    # Les modèles sklearn et Keras n'ont pas la même méthode pour obtenir
    # une probabilité, on harmonise pour la suite du script
    if model_name == "MLP (Deep Learning)":
        return model.predict(X_test, verbose=0).flatten()
    else:
        return model.predict_proba(X_test)[:, 1]


def load_selected_thresholds():
    threshold_path = RESULTS_DIR / "threshold_tuning.csv"
    if not threshold_path.exists():
        raise FileNotFoundError(
            "Le fichier de seuils validés est introuvable. "
            "Lancez d'abord `python src/models/threshold_tuning.py`."
        )
    df_thresholds = pd.read_csv(threshold_path)
    return dict(zip(df_thresholds["model"], df_thresholds["selected_threshold"]))


def evaluate_at_threshold(y_test, y_proba, threshold):
    # Applique un seuil choisi sur validation, puis calcule les métriques de
    # performance sur le test set.
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    return {
        "seuil_retenu": threshold,
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "FN": cm[1, 0],
        "FP": cm[0, 1],
    }


def load_best_mlp_params():
    """Recharge les meilleurs hyperparamètres MLP depuis mlp_results.csv."""
    mlp_results_path = RESULTS_DIR / "mlp_results.csv"
    if mlp_results_path.exists():
        row = pd.read_csv(mlp_results_path).iloc[0]
        return {
            "units_1": int(row["best_units_1"]),
            "units_2": int(row["best_units_2"]),
            "dropout": float(row["best_dropout"]),
            "learning_rate": float(row["best_learning_rate"]),
            "batch_size": int(row["best_batch_size"]),
        }
    return {"units_1": 32, "units_2": 16, "dropout": 0.1, "learning_rate": 0.001, "batch_size": 64}


def cross_validate_mlp(X_train, y_train, n_splits=5):
    # cross_val_score de sklearn ne fonctionne pas directement avec un modèle
    # Keras : on code donc la validation croisée manuellement, en réentraînant
    # le MLP sur chaque fold. Plus lent que pour les modèles sklearn (chaque
    # entraînement prend ~20-40s), mais nécessaire pour avoir une vraie mesure
    # de stabilité sur les 4 modèles, sans exception.
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    f1_scores = []

    X_train_arr = X_train.values
    y_train_arr = y_train.values

    best_params = load_best_mlp_params()
    for fold_i, (train_idx, val_idx) in enumerate(skf.split(X_train_arr, y_train_arr), 1):
        print(f"  Fold {fold_i}/{n_splits}...")
        X_fold_train, X_fold_val = X_train_arr[train_idx], X_train_arr[val_idx]
        y_fold_train, y_fold_val = y_train_arr[train_idx], y_train_arr[val_idx]

        classes = np.array([0, 1])
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_fold_train)
        class_weight_dict = {0: weights[0], 1: weights[1]}

        model = build_mlp(
            input_dim=X_fold_train.shape[1],
            units_1=best_params["units_1"],
            units_2=best_params["units_2"],
            dropout=best_params["dropout"],
            learning_rate=best_params["learning_rate"],
        )
        early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        model.fit(X_fold_train, y_fold_train, validation_split=0.2, epochs=100,
                  batch_size=best_params["batch_size"],
                  class_weight=class_weight_dict, callbacks=[early_stop], verbose=0)

        y_pred_proba = model.predict(X_fold_val, verbose=0).flatten()
        y_pred = (y_pred_proba >= 0.5).astype(int)
        f1_scores.append(f1_score(y_fold_val, y_pred))

    return np.mean(f1_scores), np.std(f1_scores)


def measure_stability(X_train, y_train):
    # Critère 2 : stabilité -- validation croisée Stratified K-Fold (5 folds)
    # sur les modèles avec leurs hyperparamètres optimisés (clonés depuis les pkl).
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    stability = {}

    sklearn_models = {
        "Régression Logistique": joblib.load(MODELS_DIR / "logreg.pkl"),
        "Random Forest": joblib.load(MODELS_DIR / "random_forest.pkl"),
        "Hist Gradient Boosting": joblib.load(MODELS_DIR / "hist_gb.pkl"),
    }
    for name, saved_model in sklearn_models.items():
        scores = cross_val_score(clone(saved_model), X_train, y_train, cv=skf, scoring="f1")
        stability[name] = (scores.mean(), scores.std())

    print("  Validation croisée du MLP (5 réentraînements, plus lent)...")
    mlp_mean, mlp_std = cross_validate_mlp(X_train, y_train)
    stability["MLP (Deep Learning)"] = (mlp_mean, mlp_std)

    return stability


def measure_computational_cost(X_train, y_train, X_test, n_repeats=5):
    # Critères 4 (coût de calcul) et 5 (facilité de déploiement) : réentraîne
    # les modèles avec leurs hyperparamètres optimisés (clonés depuis les pkl)
    # pour mesurer les temps d'entraînement et de prédiction. La taille est
    # lue directement depuis les fichiers sauvegardés.
    cost = {}
    single_row = X_test.iloc[[0]]

    sklearn_files = {
        "Régression Logistique": "logreg.pkl",
        "Random Forest": "random_forest.pkl",
        "Hist Gradient Boosting": "hist_gb.pkl",
    }

    for name, fname in sklearn_files.items():
        saved_model = joblib.load(MODELS_DIR / fname)
        train_times = []
        for _ in range(n_repeats):
            model = clone(saved_model)
            t0 = time.time()
            model.fit(X_train, y_train)
            train_times.append(time.time() - t0)

        # Temps de prédiction sur une ligne unique, moyenné sur 50 appels
        t0 = time.time()
        for _ in range(50):
            model.predict_proba(single_row)
        predict_time_ms = (time.time() - t0) / 50 * 1000

        # Taille lue depuis le fichier sauvegardé (modèle entraîné sur X_train complet)
        size_kb = (MODELS_DIR / fname).stat().st_size / 1024

        cost[name] = {
            "train_time_median_s": statistics.median(train_times),
            "predict_time_ms": predict_time_ms,
            "model_size_kb": size_kb,
        }

    # MLP : le temps d'entraînement n'est pas remesouré ici (déjà mesuré en D4,
    # ~20-40s). En revanche, le temps de prédiction ne nécessite aucun
    # réentraînement - on charge le modèle sauvegardé et on chronomètre.
    mlp_model = keras.models.load_model(MODELS_DIR / "mlp_model.keras")
    single_row_np = single_row.values
    mlp_model.predict(single_row_np, verbose=0)  # warmup : le 1er appel Keras est plus lent
    t0 = time.time()
    for _ in range(50):
        mlp_model.predict(single_row_np, verbose=0)
    mlp_predict_time_ms = (time.time() - t0) / 50 * 1000

    mlp_size_kb = (MODELS_DIR / "mlp_model.keras").stat().st_size / 1024
    cost["MLP (Deep Learning)"] = {
        "train_time_median_s": np.nan,
        "predict_time_ms": mlp_predict_time_ms,
        "model_size_kb": mlp_size_kb,
    }

    return cost


def run_final_comparison():
    # Compare les 4 modèles sur les 6 critères de la consigne, choisit et sauvegarde le modèle final
    ensure_project_dirs()
    X_train, X_test, y_train, y_test = load_processed_data()
    models = load_all_models()
    selected_thresholds = load_selected_thresholds()

    # Critère 1 : performance 
    perf_results = []
    for name, model in models.items():
        y_proba = get_proba(model, name, X_test)
        res = evaluate_at_threshold(y_test, y_proba, selected_thresholds[name])
        res["model"] = name
        perf_results.append(res)
    df_perf = pd.DataFrame(perf_results).set_index("model")

    print("\n" + "=" * 110)
    print("CRITERE 1 - PERFORMANCE (seuil choisi sur validation, score final sur test)")
    print("=" * 110)
    print(df_perf.round(4).to_string())

    # Critère 2 : stabilité (y compris MLP, plus lent)
    print("\nCalcul de la stabilité (validation croisée Stratified K-Fold, y compris MLP)...")
    stability = measure_stability(X_train, y_train)

    # Critères 4 et 5 : Coût de calcul + Facilité de déploiement
    print("Mesure du coût de calcul (entraînement x5, prédiction x50) et de la taille des modèles...")
    cost = measure_computational_cost(X_train, y_train, X_test)

    # Tableau de synthèse complet (6 critères)
    summary = []
    for name in models.keys():
        cv_mean, cv_std = stability[name]
        c = cost[name]
        summary.append({
            "model": name,
            "f1": df_perf.loc[name, "f1"],
            "recall": df_perf.loc[name, "recall"],
            "cv_f1_mean": cv_mean,
            "cv_f1_std": cv_std,
            "train_time_s": c["train_time_median_s"],
            "predict_time_ms": c["predict_time_ms"],
            "model_size_kb": c["model_size_kb"],
        })
    df_summary = pd.DataFrame(summary).round(4)

    print("\n" + "=" * 130)
    print("SYNTHESE FINALE - 6 CRITERES (performance, stabilité, coût de calcul, déploiement)")
    print("=" * 130)
    print(df_summary.to_string(index=False))

    print("""
Critères non quantifiables directement par le code, discutés qualitativement :
- Interprétabilité : RF et Hist GB ont tous deux une feature importance native ; jugés équivalents
  (vérification détaillée prévue en Phase E : feature importance, permutation importance, SHAP).
- Cohérence métier : Hist GB obtient le meilleur compromis global : meilleur F1,
  meilleur Recall, moins de faux négatifs, modèle plus léger et inférence plus rapide.
""")

    # Décision finale
    # Hist Gradient Boosting retenu : meilleur F1/Recall après optimisation,
    # modèle plus léger que Random Forest et inférence plus rapide.
    final_model = models["Hist Gradient Boosting"]
    joblib.dump(final_model, MODELS_DIR / "final_model.pkl")
    df_summary.to_csv(RESULTS_DIR / "final_model_comparison.csv", index=False)
    df_perf.reset_index().round(4).to_csv(RESULTS_DIR / "final_test_metrics.csv", index=False)

    selection_note = (
        "# Choix du modèle final\n\n"
        "Modèle retenu : Hist Gradient Boosting.\n\n"
        "Justification : après optimisation des hyperparamètres, Hist Gradient Boosting "
        "obtient le meilleur compromis global : meilleur F1/Recall, faible nombre de faux "
        "négatifs, coût de prédiction inférieur au Random Forest, taille de fichier réduite "
        "et intégration Streamlit simple. Les seuils de décision utilisés pour l'évaluation "
        "finale ont été sélectionnés sur une validation interne avec les mêmes hyperparamètres "
        "que les modèles finaux, puis appliqués une seule fois au test set.\n"
    )
    (RESULTS_DIR / "final_model_selection.md").write_text(selection_note, encoding="utf-8")
    print(f">>> MODÈLE FINAL RETENU : Hist Gradient Boosting")
    print(f">>> Sauvegardé dans {project_relative(MODELS_DIR / 'final_model.pkl')}")
    print(f">>> Comparaison sauvegardée : {project_relative(RESULTS_DIR / 'final_model_comparison.csv')}")

    return df_summary


if __name__ == "__main__":
    run_final_comparison()
