# Modélisation : comparaison des 3 modèles ML

# Objectif : entraîner et comparer 3 modèles de machine learning
# avec une validation croisée Stratified K-Fold
# On retient class_weight="balanced" pour les 3 modèles : les 3 supportent ce paramètre nativement, donc on peut les
# traiter de la même façon (même pondération, même validation croisée).

# Modèles comparés :
# régression logistique (linéaire, très interprétable), Random Forest et Hist Gradient Boosting
import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
import time

PROCESSED_DIR = "data/processed"
MODELS_DIR = "outputs/models"


def load_processed_data():
    # Recharge les données déjà préparées par le pipeline de preprocessing
    X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv")
    X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
    y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train.csv").squeeze()
    y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv").squeeze()
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
    # validation croisée Stratified K-Fold (5 folds) : à chaque découpage, la
    # proportion de pannes (14.8%) est préservée dans chaque fold, contrairement
    # à un K-Fold classique qui pourrait créer des folds très différents les uns
    # des autres sur un dataset déséquilibré
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="f1")
    return scores.mean(), scores.std()


def run_ml_comparison():
    # entraîne et compare les 3 modèles ML, tous avec class_weight='balanced'
    X_train, X_test, y_train, y_test = load_processed_data()

    results = []

    # 1. Régression logistique
    print("[1/3] Régression logistique...")
    t0 = time.time()
    model_lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    cv_mean, cv_std = cross_validate_model(model_lr, X_train, y_train)
    model_lr.fit(X_train, y_train)
    train_time = time.time() - t0
    res = evaluate(model_lr, X_test, y_test)
    res.update({"model": "Régression Logistique", "cv_f1_mean": cv_mean, "cv_f1_std": cv_std, "train_time_s": train_time})
    results.append(res)

    # 2. Random Forest
    print("[2/3] Random Forest...")
    t0 = time.time()
    model_rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight="balanced", n_jobs=-1)
    cv_mean, cv_std = cross_validate_model(model_rf, X_train, y_train)
    model_rf.fit(X_train, y_train)
    train_time = time.time() - t0
    res = evaluate(model_rf, X_test, y_test)
    res.update({"model": "Random Forest", "cv_f1_mean": cv_mean, "cv_f1_std": cv_std, "train_time_s": train_time})
    results.append(res)

    # 3. Hist Gradient Boosting (supporte nativement class_weight, contrairement
    # à GradientBoostingClassifier classique)
    t0 = time.time()
    model_gb = HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=0.1,
                                                random_state=42, class_weight="balanced")
    cv_mean, cv_std = cross_validate_model(model_gb, X_train, y_train)
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
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model_lr, f"{MODELS_DIR}/logreg.pkl")
    joblib.dump(model_rf, f"{MODELS_DIR}/random_forest.pkl")
    joblib.dump(model_gb, f"{MODELS_DIR}/hist_gb.pkl")
    print(f"\nModèles sauvegardés dans {MODELS_DIR}/ (logreg.pkl, random_forest.pkl, hist_gb.pkl)")

    return df_results, {"logreg": model_lr, "rf": model_rf, "gb": model_gb}


if __name__ == "__main__":
    run_ml_comparison()