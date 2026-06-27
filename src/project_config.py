from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent

DATASET_FILENAME = "predictive_maintenance_v3.csv"
DATASET_SOURCE_URL = (
    "https://www.kaggle.com/datasets/tatheerabbas/"
    "industrial-machine-predictive-maintenance?resource=download"
)

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_PATH = RAW_DATA_DIR / DATASET_FILENAME
ARCHIVE_DATA_PATH = WORKSPACE_ROOT / "archive" / DATASET_FILENAME

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"


def ensure_project_dirs() -> None:
    for directory in (RAW_DATA_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, RESULTS_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def resolve_raw_data_path() -> Path:
    """Return the first available local copy of the raw Kaggle dataset."""
    candidates = (RAW_DATA_PATH, ARCHIVE_DATA_PATH)
    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(
        "Dataset brut introuvable.\n"
        f"Fichier attendu : {DATASET_FILENAME}\n"
        f"Emplacements recherches :\n{searched}\n"
        f"Source Kaggle : {DATASET_SOURCE_URL}\n"
        "Placez le CSV dans data/raw/ ou dans ../archive/ puis relancez."
    )


def project_relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)
