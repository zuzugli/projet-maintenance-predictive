# Interprétabilité du modèle final (Hist Gradient Boosting)

# Objectif : expliquer pourquoi le modèle prend ses décisions, en
# langage compréhensible pour un utilisateur non spécialiste. Techniques utilisées :
# - feature_importances_ natif : non disponible pour HistGradientBoostingClassifier
# - Permutation Importance
# - SHAP 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap
from sklearn.inspection import permutation_importance
from src.project_config import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, RESULTS_DIR, ensure_project_dirs, project_relative


def load_processed_data():
    # Recharge les données déjà préparées par le pipeline de preprocessing
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze()
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def run_permutation_importance(model, X_test, y_test):
    # Permutation Importance. Principe :
    # 1. Mesurer la performance initiale du modèle (ici, le F1)
    # 2. Mélanger aléatoirement une variable (en gardant les autres intactes)
    # 3. Recalculer la performance.
    # 4. Observer la perte de performance : plus elle chute, plus la variable est importante 
    # On répète l'opération n_repeats fois par variable pour une mesure stable
    result = permutation_importance(
        model, X_test, y_test, scoring="f1", n_repeats=10, random_state=42, n_jobs=-1
    )
    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)
    return importance_df


def plot_permutation_importance(importance_df):
    # Visualise les variables les plus importantes
    ensure_project_dirs()
    top_n = importance_df.head(10)
    plt.figure(figsize=(9, 6))
    plt.barh(top_n["feature"][::-1], top_n["importance_mean"][::-1],
              xerr=top_n["importance_std"][::-1], color="steelblue")
    plt.xlabel("Diminution du F1-score quand la variable est mélangée")
    plt.title("Permutation Importance - Top 10 variables (Hist Gradient Boosting)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "05_permutation_importance.png", dpi=120)
    plt.close()


def run_shap_analysis(model, X_test):
    # SHAP (SHapley Additive exPlanations) : explique chaque prédiction
    # individuelle en répartissant la contribution de chaque variable à la
    # prédiction finale. On utilise ici un échantillon du test (100 lignes)
    # plutôt que tout le test set, car le calcul SHAP est coûteux
    X_sample = X_test.sample(n=100, random_state=42)
    explainer = shap.TreeExplainer(model, X_sample)
    shap_values = explainer(X_sample, check_additivity=False)

    # Graphique global : importance moyenne (en valeur absolue) de chaque variable
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "06_shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close()

    return shap_values, X_sample


def run_interpretability():
    # Charge le modèle final et les données, calcule et visualise Permutation Importance + SHAP
    ensure_project_dirs()
    X_train, X_test, y_train, y_test = load_processed_data()
    model = joblib.load(MODELS_DIR / "final_model.pkl")

    print("=" * 80)
    print("PERMUTATION IMPORTANCE")
    print("=" * 80)
    importance_df = run_permutation_importance(model, X_test, y_test)
    print(importance_df.to_string(index=False))
    importance_df.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)
    plot_permutation_importance(importance_df)
    print(f"\nImportance sauvegardée : {project_relative(RESULTS_DIR / 'feature_importance.csv')}")
    print(f"Graphique sauvegardé : {project_relative(FIGURES_DIR / '05_permutation_importance.png')}")

    print("\n" + "=" * 80)
    print("ANALYSE SHAP")
    print("=" * 80)
    print("Calcul des valeurs SHAP sur un échantillon de 100 lignes du test...")
    shap_values, X_sample = run_shap_analysis(model, X_test)
    print(f"Graphique sauvegardé : {project_relative(FIGURES_DIR / '06_shap_summary.png')}")

    return importance_df, shap_values


if __name__ == "__main__":
    run_interpretability()
