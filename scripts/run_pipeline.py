import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.baseline import run_baseline
from src.models.final_comparison import run_final_comparison
from src.models.interpretability import run_interpretability
from src.models.rebalancing import run_rebalancing_comparison
from src.models.threshold_tuning import run_threshold_tuning
from src.models.train_ml_models import run_ml_comparison
from src.models.train_mlp import run_mlp
from src.preprocessing.pipeline import run_preprocessing
from src.project_config import project_relative, resolve_raw_data_path


def main():
    raw_path = resolve_raw_data_path()
    print(f"Dataset brut utilise : {project_relative(raw_path)}")

    steps = [
        ("Preprocessing", run_preprocessing),
        ("Baseline", run_baseline),
        ("Reequilibrage", run_rebalancing_comparison),
        ("Modeles ML", run_ml_comparison),
        ("Modele MLP", run_mlp),
        ("Tuning du seuil", run_threshold_tuning),
        ("Comparaison finale", run_final_comparison),
        ("Interpretabilite", run_interpretability),
    ]

    for label, fn in steps:
        print("\n" + "=" * 100)
        print(label.upper())
        print("=" * 100)
        fn()

    print("\nPipeline complet termine. Artefacts disponibles dans outputs/models et outputs/results.")


if __name__ == "__main__":
    main()
