# Maintenance Prédictive Industrielle - Projet Data Science M1

Prédiction de pannes machines dans les 24h à partir de données capteurs, avec comparaison de 4 modèles et un dashboard décisionnel interactif.

---

## Problème

Données capteurs temps réel (vibration, température, courant, pression, RPM...) issues de 4 types de machines industrielles. L'objectif est de prédire si une panne surviendra dans les 24h pour permettre une maintenance préventive.

Dataset : `data/raw/predictive_maintenance_v3.csv` (~24 000 enregistrements, ~19% de pannes)

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
│       └── dashboard/
│           └── app.py          # Dashboard Streamlit
├── outputs/
│   ├── models/                 # Modèles sauvegardés (.pkl, .keras)
│   └── figures/                # Graphiques générés (dont SHAP)
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Lancer le dashboard

```bash
streamlit run src/dashboard/app.py
```

---

## Pipeline d'entraînement (ordre d'exécution)

```bash
python src/preprocessing/pipeline.py
python src/models/train_ml_models.py
python src/models/train_mlp.py
python src/models/threshold_tuning.py
python src/models/final_comparison.py
python src/models/interpretability.py
```

---

## Résultats

| Modèle | F1 | Recall | ROC-AUC | Seuil retenu |
|---|---|---|---|---|
| Régression Logistique | 0.766 | 0.806 | 0.959 | 0.70 |
| Random Forest | 0.863 | 0.899 | 0.989 | 0.65 |
| **Hist Gradient Boosting (retenu)** | **0.862** | **0.889** | **0.987** | **0.75** |
| MLP (Deep Learning) | 0.821 | 0.843 | 0.978 | 0.80 |

**Modèle retenu : Hist Gradient Boosting**
Performance quasi identique au Random Forest, mais 40x plus léger (213 Ko vs 8,5 Mo) et 7x plus rapide en prédiction (2,12 ms vs 14,39 ms).

### Variables les plus importantes
1. Vitesse de rotation (RPM) - 41%
2. Température moteur - 26%
3. Courant électrique - 21%
4. Vibration (RMS) - 19%

---

## Dashboard

4 onglets :
- **Prédiction en temps réel** - saisie d'un scénario capteur, estimation du risque de panne avec jauge
- **Comparaison des modèles** - métriques, matrices de confusion, bilan écoresponsabilité
- **Importance des variables** - Permutation Importance + SHAP
- **Exploration des données** - distributions, boxplots, heatmap de corrélations
