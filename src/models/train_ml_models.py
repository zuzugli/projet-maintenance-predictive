# Modélisation : comparaison des 3 modèles ML

# Objectif : entraîner et comparer 3 modèles de machine learning
# avec optimisation des hyperparamètres (RandomizedSearchCV) et
# validation croisée Stratified K-Fold.

# Modèles comparés :
# régression logistique (linéaire, très interprétable), Random Forest et Hist Gradient Boosting
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
import time
from src.project_config import MODELS_DIR, PROCESSED_DIR, RESULTS_DIR, ensure_project_dirs, project_relative


def load_processed_data():
    # Recharge les données déjà préparées par le pipeline de preprocessing
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze()
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def evaluate(model, X_test, y_test):
    # Calcule les métriques importantes
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


def cross_validate_model(model, X_train, y_train):
    # Validation croisée Stratified K-Fold (5 folds) : la proportion de pannes
    # est préservée dans chaque fold, essentiel sur un dataset déséquilibré.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="f1")
    return scores.mean(), scores.std()


def tune_model(estimator, param_distributions, X_train, y_train, n_iter=20):
    # RandomizedSearchCV : explore n_iter combinaisons aléatoires dans l'espace
    # des hyperparamètres. Plus rapide que GridSearchCV pour les grands espaces,
    # avec des résultats comparables. Scoring = f1 car dataset déséquilibré.
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator,
        param_distributions,
        n_iter=n_iter,
        scoring="f1",
        cv=skf,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    print(f"    Meilleurs params : {search.best_params_}  |  F1 CV : {search.best_score_:.4f}")
    return search.best_estimator_


def run_ml_comparison():
    # Entraîne et compare les 3 modèles ML avec optimisation des hyperparamètres
    ensure_project_dirs()
    X_train, X_test, y_train, y_test = load_processed_data()

    results = []

    # 1. Régression logistique
    print("[1/3] Régression logistique - optimisation des hyperparamètres...")
    lr_params = {
        "C": [0.01, 0.1, 1, 10, 100],
        "solver": ["lbfgs", "liblinear"],
        "max_iter": [500, 1000, 2000],
    }
    model_lr = tune_model(
        LogisticRegression(random_state=42, class_weight="balanced"),
        lr_params, X_train, y_train,
    )
    cv_mean, cv_std = cross_validate_model(model_lr, X_train, y_train)
    t0 = time.time()
    model_lr.fit(X_train, y_train)
    train_time = time.time() - t0
    res = evaluate(model_lr, X_test, y_test)
    res.update({"model": "Régression Logistique", "cv_f1_mean": cv_mean, "cv_f1_std": cv_std, "train_time_s": train_time})
    results.append(res)

    # 2. Random Forest
    print("[2/3] Random Forest - optimisation des hyperparamètres...")
    rf_params = {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    model_rf = tune_model(
        RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1),
        rf_params, X_train, y_train,
    )
    cv_mean, cv_std = cross_validate_model(model_rf, X_train, y_train)
    t0 = time.time()
    model_rf.fit(X_train, y_train)
    train_time = time.time() - t0
    res = evaluate(model_rf, X_test, y_test)
    res.update({"model": "Random Forest", "cv_f1_mean": cv_mean, "cv_f1_std": cv_std, "train_time_s": train_time})
    results.append(res)

    # 3. Hist Gradient Boosting
    print("[3/3] Hist Gradient Boosting - optimisation des hyperparamètres...")
    gb_params = {
        "max_iter": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
        "min_samples_leaf": [20, 50, 100],
    }
    model_gb = tune_model(
        HistGradientBoostingClassifier(random_state=42, class_weight="balanced"),
        gb_params, X_train, y_train,
    )
    cv_mean, cv_std = cross_validate_model(model_gb, X_train, y_train)
    t0 = time.time()
    model_gb.fit(X_train, y_train)
    train_time = time.time() - t0
    res = evaluate(model_gb, X_test, y_test)
    res.update({"model": "Hist Gradient Boosting", "cv_f1_mean": cv_mean, "cv_f1_std": cv_std, "train_time_s": train_time})
    results.append(res)

    # Tableau récapitulatif
    df_results = pd.DataFrame(results)[
        ["model", "recall", "f1", "roc_auc", "pr_auc", "FN", "FP", "cv_f1_mean", "cv_f1_std", "train_time_s"]
    ].round(4)

    print("\n" + "=" * 100)
    print("COMPARAISON DES 3 MODELES ML (tous avec correction du déséquilibre)")
    print("=" * 100)
    print(df_results.to_string(index=False))

    # Sauvegarde des 3 modèles entraînés, pour être réutilisés en D6 (synthèse
    # finale) sans avoir à les réentraîner depuis zéro.
    joblib.dump(model_lr, MODELS_DIR / "logreg.pkl")
    joblib.dump(model_rf, MODELS_DIR / "random_forest.pkl")
    joblib.dump(model_gb, MODELS_DIR / "hist_gb.pkl")
    df_results.to_csv(RESULTS_DIR / "ml_models_comparison.csv", index=False)
    print(f"\nModèles sauvegardés dans {project_relative(MODELS_DIR)}/ (logreg.pkl, random_forest.pkl, hist_gb.pkl)")
    print(f"Résultats sauvegardés : {project_relative(RESULTS_DIR / 'ml_models_comparison.csv')}")

    return df_results, {"logreg": model_lr, "rf": model_rf, "gb": model_gb}


if __name__ == "__main__":
    run_ml_comparison()
