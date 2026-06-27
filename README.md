# Maintenance Prédictive Industrielle - Projet Data Science M1

Prédiction de pannes machines dans les 24h à partir de données capteurs, avec comparaison de 4 modèles et un dashboard décisionnel interactif.

---

## Problème

Données capteurs temps réel (vibration, température, courant, pression, RPM...) issues de 4 types de machines industrielles. L'objectif est de prédire si une panne surviendra dans les 24h pour permettre une maintenance préventive.

Dataset : `data/raw/predictive_maintenance_v3.csv` (24 042 enregistrements, 14,8% de pannes).

Source Kaggle : <https://www.kaggle.com/datasets/tatheerabbas/industrial-machine-predictive-maintenance?resource=download>

---

## Structure du projet

```
├── data/
│   ├── raw/                    # Dataset brut
│   └── processed/              # Données préparées (X_train, X_test, y_train, y_test)
├── notebooks/
│   └── 01_eda.ipynb            # Analyse exploratoire
├── src/
│   ├── preprocessing/
│   │   └── pipeline.py         # Pipeline de preprocessing (encodage, normalisation, split)
│   └── models/
│       ├── baseline.py         # Modèle de référence (régression logistique sans réglage)
│       ├── rebalancing.py      # Comparaison des stratégies de gestion du déséquilibre
│       ├── train_ml_models.py  # Entraînement : Régression Logistique, Random Forest, Hist GB
│       ├── train_mlp.py        # Entraînement : MLP (Deep Learning)
│       ├── threshold_tuning.py # Ajustement du seuil de décision (maximise le F1)
│       ├── final_comparison.py # Comparaison finale 6 critères + sauvegarde du modèle retenu
│       ├── interpretability.py # Permutation Importance + SHAP
├── src/dashboard/
│   └── app.py                  # Dashboard Streamlit
├── scripts/
│   └── run_pipeline.py         # Régénération complète des artefacts
├── outputs/
│   ├── models/                 # Modèles sauvegardés (.pkl, .keras)
│   ├── results/                # Métriques et comparaisons exportées (.csv)
│   └── figures/                # Graphiques générés (dont SHAP)
├── reports/
│   └── rapport_projet.md       # Rapport analytique structuré
└── requirements.txt
```

---

## Rapport

Le rapport analytique complet est disponible ici : [reports/rapport_projet.md](reports/rapport_projet.md)

---

## Installation

Python recommandé : 3.10 à 3.12. TensorFlow n'est pas encore compatible avec tous les environnements Python très récents.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Lancer le dashboard

Si les artefacts `outputs/models/preprocessor.pkl` et `outputs/models/final_model.pkl`
ne sont pas encore présents, lancez d'abord le pipeline complet.

```bash
streamlit run src/dashboard/app.py
```

---

## Pipeline complet reproductible

Commande recommandée :

```bash
python scripts/run_pipeline.py
```

Le script lit automatiquement le dataset depuis :
1. `data/raw/predictive_maintenance_v3.csv` si présent ;
2. `../archive/predictive_maintenance_v3.csv` en fallback local.

Ordre d'exécution détaillé :

```bash
python -m src.preprocessing.pipeline
python -m src.models.baseline
python -m src.models.rebalancing
python -m src.models.train_ml_models
python -m src.models.train_mlp
python -m src.models.threshold_tuning
python -m src.models.final_comparison
python -m src.models.interpretability
```

Les fichiers de résultats sont exportés dans `outputs/results/` :
- `baseline_results.csv`
- `rebalancing_comparison.csv`
- `ml_models_comparison.csv`
- `mlp_results.csv`
- `threshold_tuning.csv`
- `final_model_comparison.csv`
- `final_test_metrics.csv`
- `feature_importance.csv`

---

## Résultats

| Modèle | F1 | Recall | ROC-AUC | Seuil retenu |
|---|---|---|---|---|
| Régression Logistique | 0.765 | 0.862 | 0.959 | 0.60 |
| Random Forest | 0.890 | 0.928 | 0.994 | 0.55 |
| **Hist Gradient Boosting (retenu)** | **0.908** | **0.944** | **0.996** | **0.60** |
| MLP (Deep Learning) | 0.817 | 0.903 | 0.981 | 0.70 |

Les seuils sont sélectionnés sur un split de validation interne du train set, puis évalués une seule fois sur le test set. Les hyperparamètres sont optimisés par RandomizedSearchCV (modèles sklearn) et recherche aléatoire manuelle (MLP).

**Modèle retenu : Hist Gradient Boosting**
Après optimisation des hyperparamètres, HistGB est le meilleur modèle sur tous les critères : F1 = 0.908 (vs 0.890 pour RF), 40x plus léger (213 Ko vs 8 531 Ko) et 13x plus rapide en prédiction (4.03 ms vs 54.05 ms).

### Variables les plus importantes
1. Vitesse de rotation (RPM) - 40%
2. Température moteur - 31%
3. Vibration (RMS) - 17%
4. Courant électrique - 16%

---

## Dashboard

4 onglets :
- **Prédiction en temps réel** - saisie d'un scénario capteur, estimation du risque de panne avec jauge
- **Comparaison des modèles** - métriques, matrices de confusion, bilan écoresponsabilité
- **Importance des variables** - Permutation Importance + SHAP
- **Exploration des données** - distributions, boxplots, heatmap de corrélations

Le dashboard charge en priorité les métriques générées dans `outputs/results/`. Si elles ne sont pas encore présentes, il conserve des valeurs de démonstration.
