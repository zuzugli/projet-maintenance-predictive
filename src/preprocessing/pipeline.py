import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Chemins
RAW_DATA_PATH = "data/raw/predictive_maintenance_v3.csv"
PROCESSED_DIR = "data/processed"
MODELS_DIR = "outputs/models"

# Variables gardées (voir 01_eda.ipynb).
# Exclues : timestamp/machine_id (identifiants bruts) ;
# rul_hours/failure_type/estimated_repair_cost (fuites de données : connues
# seulement pendant/après la panne).
FEATURE_COLS = [
    "machine_type", "vibration_rms", "temperature_motor", "current_phase_avg",
    "pressure_level", "rpm", "operating_mode", "hours_since_maintenance", "ambient_temp",
]
TARGET_COL = "failure_within_24h"

NUMERIC_FEATURES = [
    "vibration_rms", "temperature_motor", "current_phase_avg",
    "pressure_level", "rpm", "hours_since_maintenance", "ambient_temp",
]
CATEGORICAL_FEATURES = ["machine_type", "operating_mode"]


def load_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    # Charge le CSV brut
    df = pd.read_csv(path)
    print(f"[load_data] Shape brute : {df.shape}")
    return df


def select_features(df: pd.DataFrame):
    # Garde uniquement les colonnes utiles à la prédiction (FEATURE_COLS) et
    #  isole la colonne cible (TARGET_COL)
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    print(f"[select_features] X : {X.shape} | y : {y.shape}")
    print(f"[select_features] Distribution cible :\n{y.value_counts(normalize=True)}")
    return X, y


def build_preprocessor() -> ColumnTransformer:
    # Construit le préprocesseur :
    #- Sous-pipeline numérique : remplit les valeurs manquantes par la médiane
    # (robuste aux outliers), puis met toutes les variables à la même échelle
    # avec StandardScaler 
    #- Sous-pipeline catégoriel : transforme chaque catégorie texte en colonnes
    #  0/1 avec OneHotEncoder
    #- ColumnTransformer applique chaque sous-pipeline au bon groupe de colonnes
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
    return preprocessor


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    # Découpe les données en train (80%) et test (20%)
    # stratify=y force le même pourcentage de pannes dans le train et le test
    # random_state=42 fixe le hasard : on obtient le même découpage à chaque exécution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"[split_data] Train : {X_train.shape[0]} lignes | Test : {X_test.shape[0]} lignes")
    print(f"[split_data] Proportion pannes - train : {y_train.mean():.4f} | test : {y_test.mean():.4f}")
    return X_train, X_test, y_train, y_test


def run_preprocessing():
    # fonction principale 
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = load_data()
    X, y = select_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    preprocessor = build_preprocessor()

    # fit_transform() sur le train : le préprocesseur apprend ses paramètres
    # (médianes, moyennes/écarts-types, catégories) uniquement à partir du train.
    # transform() sur le test (sans fit) : on réapplique ces mêmes paramètres,
    # sans rien recalculer à partir du test. C'est ce qui évite le data leakage.
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()
    print(f"[run_preprocessing] Features après preprocessing ({len(feature_names)}) : {list(feature_names)}")

    # On remet les tableaux numpy en DataFrame avant de sauvegarder en CSV.
    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)

    X_train_df.to_csv(f"{PROCESSED_DIR}/X_train.csv", index=False)
    X_test_df.to_csv(f"{PROCESSED_DIR}/X_test.csv", index=False)
    y_train.to_csv(f"{PROCESSED_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{PROCESSED_DIR}/y_test.csv", index=False)

    # On sauvegarde aussi le préprocesseur déjà entraîné (pas juste les données)
    # il sera réutilisé plus tard pour transformer une nouvelle donnée saisie
    # dans le dashboard ou reçue par l'API, exactement de la même façon.
    joblib.dump(preprocessor, f"{MODELS_DIR}/preprocessor.pkl")

    print(f"[run_preprocessing] Fichiers sauvegardés dans {PROCESSED_DIR}/ et {MODELS_DIR}/")
    return X_train_df, X_test_df, y_train, y_test


if __name__ == "__main__":
    run_preprocessing()