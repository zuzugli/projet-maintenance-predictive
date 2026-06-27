# Modélisation : comparaison des techniques de rééquilibrage

# Objectif : comparer au moins 2 techniques de rééquilibrage data-level
# (parmi Random Over-Sampling, SMOTE, Random Under-Sampling), plus l'approche
# niveau-modèle class_weight="balanced", toutes face au baseline

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from src.project_config import PROCESSED_DIR, RESULTS_DIR, ensure_project_dirs, project_relative


def load_processed_data():
    # Recharge les données déjà préparées par le pipeline de preprocessing
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze()
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def evaluate(model, X_test, y_test):
    # Calcule les métriques importantes pour comparer les techniques entre elles
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    return {
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "FN": cm[1, 0],
        "FP": cm[0, 1],
    }


def run_rebalancing_comparison():
    # entraîne la même régression logistique avec différentes stratégies de
    # rééquilibrage, et compare les résultats dans un tableau récapitulatif
    ensure_project_dirs()
    X_train, X_test, y_train, y_test = load_processed_data()

    results = []

    # 1. Baseline
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    res = evaluate(model, X_test, y_test)
    res["technique"] = "Baseline (aucun rééquilibrage)"
    results.append(res)

    # 2. class_weight="balanced" : approche niveau-modèle, ne touche pas aux données,
    # juste pénalise plus fortement les erreurs sur la classe minoritaire pendant l'entraînement
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    res = evaluate(model, X_test, y_test)
    res["technique"] = "class_weight='balanced'"
    results.append(res)

    # 3. Random Over-Sampling : duplique des exemples de la classe minoritaire
    # (pannes) jusqu'à équilibrer les classes dans le train
    ros = RandomOverSampler(random_state=42)
    X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_ros, y_train_ros)
    res = evaluate(model, X_test, y_test)
    res["technique"] = "Random Over-Sampling"
    results.append(res)

    # 4. SMOTE : crée de nouveaux exemples synthétiques de pannes
    # plutôt que de dupliquer les mêmes lignes comme Random Over-Sampling
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_smote, y_train_smote)
    res = evaluate(model, X_test, y_test)
    res["technique"] = "SMOTE"
    results.append(res)

    # 5. Random Under-Sampling : supprime des exemples de la classe majoritaire
    # (pas de panne) jusqu'à équilibrer les classes
    rus = RandomUnderSampler(random_state=42)
    X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_rus, y_train_rus)
    res = evaluate(model, X_test, y_test)
    res["technique"] = "Random Under-Sampling"
    results.append(res)

    # Tableau récapitulatif
    df_results = pd.DataFrame(results)[["technique", "recall", "f1", "roc_auc", "pr_auc", "FN", "FP"]]
    df_results = df_results.round(4)
    print("\n" + "=" * 90)
    print("COMPARAISON DES TECHNIQUES DE REEQUILIBRAGE (modèle : régression logistique)")
    print("=" * 90)
    print(df_results.to_string(index=False))
    print(f"\nRappel : sur le test (4809 lignes), il y a 712 vraies pannes au total.")
    print("FN = pannes manquées (le plus coûteux en contexte industriel) | FP = fausses alertes")
    df_results.to_csv(RESULTS_DIR / "rebalancing_comparison.csv", index=False)
    print(f"Résultats sauvegardés : {project_relative(RESULTS_DIR / 'rebalancing_comparison.csv')}")

    return df_results


if __name__ == "__main__":
    run_rebalancing_comparison()
