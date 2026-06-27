# Rapport Projet Data Science

## Page de garde

**Titre du projet :** Système intelligent multi-modèles pour la maintenance prédictive industrielle

**Promotion :** Mastère Data Engineering et IA - M1

**Année scolaire :** 2025-2026

**Auteurs :** Zuzanna GLINIAK - Tristan BOURHIS

**Formatrice :** Sarah MALAEB

**Date du rapport :** 27/06/2026

**Mention obligatoire :** Projet certifiant RNCP40875 Expert en Ingénierie de Données - Bloc 2 : Pilotage et implémentation de solutions IA.

---

## 1. Résumé exécutif

Ce projet vise à construire une solution de maintenance prédictive industrielle capable d'anticiper les pannes machines dans les 24 heures à partir de données capteurs. Le besoin métier est de permettre à un responsable maintenance de prioriser les interventions, réduire les arrêts non planifiés et mieux piloter les risques opérationnels.

Le dataset utilisé provient de Kaggle : [Industrial Machine Predictive Maintenance](https://www.kaggle.com/datasets/tatheerabbas/industrial-machine-predictive-maintenance?resource=download). Il contient 24 042 observations et 15 colonnes décrivant des machines industrielles, leurs mesures physiques et plusieurs variables de résultat. La tâche prédictive retenue est une classification binaire : prédire la variable `failure_within_24h`.

La démarche suivie couvre l'ensemble d'un pipeline Data Science : analyse exploratoire, préparation des données, gestion des valeurs manquantes, encodage, standardisation, entraînement de plusieurs modèles, comparaison quantitative, interprétabilité et intégration dans un dashboard Streamlit. Quatre modèles principaux sont comparés : régression logistique, Random Forest, Hist Gradient Boosting et MLP Deep Learning.

Le modèle final retenu est le **Hist Gradient Boosting**, car il offre le meilleur compromis opérationnel entre performance, rapidité de prédiction, taille du modèle et facilité de déploiement. Sur l'évaluation finale, après sélection du seuil sur validation interne, il obtient un F1-score de 0,8481, un recall de 0,9256 et un ROC-AUC de 0,9861. Le Random Forest obtient un F1-score légèrement supérieur (0,8573) et un recall légèrement supérieur (0,9284), mais il est beaucoup plus lourd et plus lent en inférence.

L'outil final proposé est un dashboard décisionnel Streamlit permettant de simuler un scénario machine, obtenir une probabilité de panne, comparer les modèles, consulter les matrices de confusion, visualiser les variables influentes et explorer les données. Une API REST n'a pas été implémentée, car elle est optionnelle dans le cahier des charges, mais elle constitue une piste d'industrialisation naturelle.

---

## 2. Introduction et contexte

La maintenance prédictive consiste à anticiper les défaillances d'équipements avant qu'elles ne provoquent un arrêt de production. Dans un contexte industriel, une panne non planifiée peut entraîner des coûts importants : interruption de chaîne, mobilisation urgente des équipes, perte de rendement, retard de livraison et usure accélérée des équipements.

Les machines modernes produisent des données continues issues de capteurs : vibration, température moteur, courant électrique, pression, vitesse de rotation ou mode de fonctionnement. Ces signaux contiennent souvent des indices avant-coureurs d'une panne. L'objectif du projet est d'exploiter ces signaux avec du Machine Learning supervisé afin de prédire si une machine présente un risque de panne dans les prochaines 24 heures.

Le Machine Learning supervisé est adapté à ce problème car les observations sont associées à une cible connue : `failure_within_24h`. Le modèle apprend une relation entre les variables explicatives et la cible, puis généralise cette relation à de nouveaux scénarios. Le projet ne se limite pas à entraîner un modèle : il construit une chaîne complète allant des données brutes à un outil d'aide à la décision.

---

## 3. Analyse du besoin utilisateur

L'utilisateur cible est un responsable maintenance ou un ingénieur industriel. Son objectif n'est pas de manipuler un notebook, mais de disposer d'un outil simple pour identifier les machines à surveiller et décider quand déclencher une intervention.

Les décisions à soutenir sont les suivantes :

- déterminer si une machine présente un risque de panne dans les 24 heures ;
- prioriser les interventions en fonction du niveau de risque ;
- comprendre quels capteurs contribuent au risque ;
- comparer les modèles pour choisir une solution fiable ;
- suivre les indicateurs clés du dataset et du modèle.

Le dashboard répond à trois scénarios d'usage principaux :

1. **Simulation d'un scénario machine** : l'utilisateur saisit des valeurs de capteurs et obtient une probabilité de panne.
2. **Analyse comparative** : l'utilisateur compare les performances des modèles et comprend pourquoi le modèle final a été retenu.
3. **Exploration et interprétation** : l'utilisateur consulte les distributions, les corrélations et les variables influentes.

Les indicateurs attendus sont la probabilité de panne, le statut de risque, les faux négatifs, les faux positifs, les métriques F1/recall/precision/ROC-AUC, ainsi que l'importance des variables.

---

## 4. Méthodologie de travail et gestion de projet

Le projet a été mené selon une démarche incrémentale proche d'un Kanban technique. Chaque étape a produit un livrable exploitable avant de passer à l'étape suivante :

1. compréhension du sujet et choix de la cible ;
2. analyse exploratoire du dataset ;
3. préparation des données ;
4. entraînement d'une baseline ;
5. gestion du déséquilibre des classes ;
6. comparaison de modèles Machine Learning ;
7. ajout d'un modèle Deep Learning ;
8. ajustement du seuil de décision ;
9. comparaison finale et choix du modèle ;
10. interprétabilité ;
11. développement du dashboard ;
12. documentation et rapport.

L'historique Git montre une progression cohérente : EDA, preprocessing, baseline, modèles ML, MLP, seuils, comparaison finale, interprétabilité, dashboard et README.

### Risques identifiés

| Risque | Impact | Réponse apportée |
|---|---:|---|
| Données manquantes sur les capteurs | Dégradation du modèle | Imputation médiane dans un pipeline |
| Déséquilibre des classes | Accuracy trompeuse | Stratified split, class weights, F1/recall/PR-AUC |
| Data leakage | Scores artificiellement élevés | Exclusion de `rul_hours`, `failure_type`, `estimated_repair_cost` |
| Seuil de décision non optimal | Trop de faux négatifs ou faux positifs | Tuning du seuil sur validation interne |
| Dashboard dépendant d'artefacts absents | Démo impossible | Script `scripts/run_pipeline.py` et exports `outputs/results/` |
| Incompatibilités de dépendances | Reproductibilité fragile | Versions figées dans `requirements.txt` |

---

## 5. Référentiel de données

### Source

Le dataset provient de Kaggle : [Industrial Machine Predictive Maintenance](https://www.kaggle.com/datasets/tatheerabbas/industrial-machine-predictive-maintenance?resource=download).

Le fichier brut utilisé est `predictive_maintenance_v3.csv`. Dans le projet, il est disponible dans `data/raw/` et peut aussi être récupéré depuis le dossier local `../archive/`.

### Volume et structure

- Nombre de lignes : 24 042
- Nombre de colonnes : 15
- Format : CSV
- Tâche retenue : classification binaire
- Variable cible : `failure_within_24h`
- Taux de panne : 14,8%

### Variables principales

| Variable | Type | Rôle métier |
|---|---|---|
| `machine_type` | Catégorielle | Type de machine industrielle |
| `vibration_rms` | Numérique | Niveau de vibration |
| `temperature_motor` | Numérique | Température moteur |
| `current_phase_avg` | Numérique | Courant électrique moyen |
| `pressure_level` | Numérique | Niveau de pression |
| `rpm` | Numérique | Vitesse de rotation |
| `operating_mode` | Catégorielle | Mode de fonctionnement |
| `hours_since_maintenance` | Numérique | Durée depuis dernière maintenance |
| `ambient_temp` | Numérique | Température ambiante |
| `failure_within_24h` | Binaire | Cible prédite |

Certaines variables disponibles ont été exclues de la modélisation afin d'éviter une fuite de données : `rul_hours`, `failure_type` et `estimated_repair_cost`. Ces variables décrivent directement ou indirectement la panne et ne seraient pas forcément disponibles avant l'événement.

---

## 6. Analyse exploratoire des données

L'EDA est réalisée dans `notebooks/01_eda.ipynb`. Elle analyse les types, valeurs manquantes, doublons, distributions, corrélations et relations entre les capteurs et la cible.

### Qualité des données

Les valeurs manquantes apparaissent uniquement sur certains capteurs :

| Variable | Valeurs manquantes |
|---|---:|
| `vibration_rms` | 1000 |
| `temperature_motor` | 834 |
| `current_phase_avg` | 731 |
| `pressure_level` | 924 |
| `rpm` | 533 |

Aucun doublon n'a été identifié. Le dataset contient 20 machines et 4 types de machines.

### Distribution de la cible

![Distribution de la cible](../outputs/figures/01_distribution_target.png)

Le dataset est déséquilibré : environ 85,2% des observations ne correspondent pas à une panne, contre 14,8% de pannes dans les 24 heures. Une simple accuracy serait donc trompeuse, car un modèle prédisant toujours "pas de panne" obtiendrait déjà une accuracy élevée tout en ratant toutes les pannes.

### Distributions des capteurs

![Distributions capteurs](../outputs/figures/02_distributions_capteurs.png)

Les variables comme `vibration_rms`, `current_phase_avg`, `pressure_level` et `rpm` présentent des distributions multimodales. Cela peut s'expliquer par la présence de plusieurs types de machines et modes de fonctionnement. La température moteur suit une distribution plus régulière autour de 50 degrés.

### Corrélations

![Heatmap de corrélations](../outputs/figures/03_heatmap_correlations.png)

Les variables les plus corrélées avec la cible sont notamment la température moteur, la vibration et les heures depuis maintenance. Les corrélations entre plusieurs capteurs montrent aussi une redondance partielle, ce qui justifie l'usage de modèles capables de gérer des interactions non linéaires.

### Boxplots capteurs / cible

![Boxplots capteurs vs cible](../outputs/figures/04_boxplots_capteurs_vs_cible.png)

Les pannes sont associées à des profils capteurs plus critiques : température plus élevée, vibration plus forte et comportements mécaniques plus extrêmes. Ces observations confirment la pertinence métier de la tâche.

---

## 7. Préparation et transformation des données

La préparation est implémentée dans `src/preprocessing/pipeline.py`.

### Choix de features

Les variables conservées sont :

- `machine_type`
- `vibration_rms`
- `temperature_motor`
- `current_phase_avg`
- `pressure_level`
- `rpm`
- `operating_mode`
- `hours_since_maintenance`
- `ambient_temp`

Les variables exclues sont :

- `timestamp` et `machine_id`, considérées comme identifiants bruts ;
- `rul_hours`, `failure_type`, `estimated_repair_cost`, considérées comme des risques de leakage.

### Pipeline de transformation

Le pipeline utilise :

- `SimpleImputer(strategy="median")` pour les variables numériques ;
- `StandardScaler` pour standardiser les variables numériques ;
- `OneHotEncoder(handle_unknown="ignore")` pour les variables catégorielles ;
- `ColumnTransformer` pour appliquer les transformations par type de variable.

La séparation train/test est stratifiée afin de conserver la proportion de pannes dans les deux ensembles. Le split utilisé est 80% train et 20% test.

Le preprocessing est ajusté uniquement sur le train set (`fit_transform`) puis appliqué au test set (`transform`). Cette stratégie limite le data leakage.

---

## 8. Pipeline IA et architecture

```mermaid
flowchart LR
    A[CSV brut Kaggle] --> B[EDA notebook]
    A --> C[Preprocessing sklearn]
    C --> D[Train/test stratifié]
    D --> E1[Régression logistique]
    D --> E2[Random Forest]
    D --> E3[Hist Gradient Boosting]
    D --> E4[MLP Deep Learning]
    E1 --> F[Comparaison métriques]
    E2 --> F
    E3 --> F
    E4 --> F
    F --> G[Tuning seuil sur validation]
    G --> H[Evaluation finale test]
    H --> I[Choix modèle final]
    I --> J[Permutation Importance et SHAP]
    I --> K[Sérialisation modèles]
    H --> L[Exports CSV outputs/results]
    J --> L
    K --> M[Dashboard Streamlit]
    L --> M
```

Le projet est organisé pour séparer les responsabilités :

- `src/preprocessing/` : préparation des données ;
- `src/models/` : entraînement, comparaison, seuils, interprétabilité ;
- `src/dashboard/` : interface Streamlit ;
- `outputs/models/` : artefacts modèles ;
- `outputs/results/` : tableaux de métriques ;
- `outputs/figures/` : graphiques ;
- `scripts/run_pipeline.py` : orchestration complète.

---

## 9. Implémentation technique

### Technologies

| Composant | Technologie |
|---|---|
| Manipulation données | pandas, numpy |
| Modélisation ML | scikit-learn |
| Déséquilibre de classes | class weights, imbalanced-learn |
| Deep Learning | TensorFlow / Keras |
| Interprétabilité | permutation importance, SHAP |
| Visualisation | matplotlib, seaborn, Plotly |
| Dashboard | Streamlit |
| Sérialisation | joblib, format Keras `.keras` |

### Modèles testés

1. **Régression logistique** : baseline interprétable et rapide.
2. **Random Forest** : modèle d'arbres robuste aux non-linéarités.
3. **Hist Gradient Boosting** : modèle performant, compact et rapide.
4. **MLP Deep Learning** : réseau neuronal dense à deux couches cachées.

Le MLP contient deux couches cachées de 32 et 16 neurones avec activation ReLU, dropout et sortie sigmoid. Un early stopping limite le surapprentissage.

---

## 10. Evaluation comparative des modèles

Les métriques retenues sont adaptées à une classification déséquilibrée :

- **Recall** : capacité à détecter les vraies pannes ;
- **F1-score** : compromis entre precision et recall ;
- **ROC-AUC** : capacité globale de séparation ;
- **PR-AUC** : utile avec classe positive minoritaire ;
- **FN/FP** : lecture opérationnelle des erreurs.

Les résultats finaux générés dans `outputs/results/final_test_metrics.csv` sont les suivants :

| Modèle | Seuil | Precision | Recall | F1 | ROC-AUC | PR-AUC | FN | FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Régression Logistique | 0,60 | 0,6861 | 0,8596 | 0,7631 | 0,9588 | 0,8376 | 100 | 280 |
| Random Forest | 0,55 | 0,7964 | 0,9284 | 0,8573 | 0,9886 | 0,9394 | 51 | 169 |
| Hist Gradient Boosting | 0,65 | 0,7827 | 0,9256 | 0,8481 | 0,9861 | 0,9280 | 53 | 183 |
| MLP Deep Learning | 0,70 | 0,6851 | 0,9199 | 0,7854 | 0,9741 | 0,8695 | 57 | 301 |

Le Random Forest obtient le meilleur F1-score (0,8573 contre 0,8481 pour Hist Gradient Boosting) et le meilleur recall (0,9284 contre 0,9256). Il manque donc deux pannes de moins sur le test set (51 FN contre 53 FN pour Hist Gradient Boosting) et produit moins de fausses alertes (169 FP contre 183 FP). Sur la performance pure, Random Forest est donc légèrement devant.

Le choix final reste cependant **Hist Gradient Boosting** pour une logique de compromis industriel. L'écart de performance est faible : 0,0092 point de F1 et 0,0028 point de recall. En contrepartie, Hist Gradient Boosting est nettement plus léger et plus rapide : 215 Ko contre 8 517 Ko pour Random Forest, et environ 2,28 ms contre 29,57 ms par prédiction. Le Random Forest est donc environ 40 fois plus lourd et 13 fois plus lent en inférence. Dans un usage de dashboard ou de service de prédiction appelé fréquemment, ce coût opérationnel compte : Hist Gradient Boosting offre une performance presque équivalente avec un déploiement plus simple, une latence plus faible et une empreinte computationnelle plus cohérente avec l'objectif d'écoresponsabilité.

Le tuning du seuil est désormais réalisé sur une validation interne du train set, puis appliqué au test set. Cette correction évite de choisir le seuil en regardant directement la performance finale du test.

---

## 11. Interprétabilité et analyse métier

L'interprétabilité est réalisée avec deux approches :

- **Permutation Importance** : mesure la perte de F1-score quand une variable est mélangée ;
- **SHAP** : explique l'impact local et global des variables sur les prédictions.

![Permutation Importance](../outputs/figures/05_permutation_importance.png)

Les variables les plus influentes sont :

1. vitesse de rotation (`rpm`) ;
2. température moteur ;
3. courant électrique ;
4. vibration ;
5. pression.

Cette hiérarchie est cohérente avec le domaine industriel : une machine soumise à des régimes élevés, une température anormale, une consommation électrique élevée et des vibrations importantes présente un risque plus élevé.

![SHAP Summary](../outputs/figures/06_shap_summary.png)

SHAP apporte une lecture plus fine : il permet de comprendre si une valeur élevée ou basse pousse la prédiction vers le risque de panne. Cette information est utile pour transformer une probabilité brute en explication exploitable par un responsable maintenance.

---

## 12. Interface utilisateur et prototype

Le dashboard est développé avec Streamlit dans `src/dashboard/app.py`. Il est organisé en quatre onglets.

### Onglet 1 : prédiction en temps réel

L'utilisateur saisit un scénario machine :

- type de machine ;
- mode opératoire ;
- vibration ;
- température moteur ;
- courant électrique ;
- pression ;
- vitesse de rotation ;
- heures depuis maintenance ;
- température ambiante.

Le dashboard transforme ces données avec le même préprocesseur que celui utilisé à l'entraînement, puis applique le modèle final. Le résultat est affiché sous forme de probabilité et de jauge de risque.

### Onglet 2 : comparaison des modèles

Cet onglet présente les métriques, les matrices de confusion et un bilan écoresponsabilité. Il explique pourquoi le modèle final est retenu.

### Onglet 3 : importance des variables

L'utilisateur visualise les variables qui influencent le plus les prédictions, ainsi qu'une figure SHAP si elle a été générée.

### Onglet 4 : exploration des données

Cet onglet permet de consulter les KPI du dataset, les distributions par classe, les boxplots et la matrice de corrélation.

Le dashboard charge en priorité les résultats exportés dans `outputs/results/`. S'ils sont absents, il conserve des valeurs de démonstration pour garder l'interface lisible, mais le pipeline complet est recommandé avant soutenance.

---

## 13. API REST

Aucune API REST n'a été implémentée dans cette version. Cette partie était optionnelle dans le sujet. L'architecture du projet prépare toutefois une extension simple : le préprocesseur et le modèle final sont sérialisés, ce qui permettrait de créer une API FastAPI avec :

- `GET /health` pour vérifier que le service est actif ;
- `POST /predict` pour recevoir un scénario machine en JSON ;
- `GET /model-info` pour exposer le modèle utilisé, le seuil et les features attendues.

Une industrialisation plus réaliste pourrait faire communiquer le dashboard avec l'API au lieu de charger directement le modèle.

---

## 14. Résultats et tests de démonstration

Deux types de scénarios peuvent être préparés pour la démonstration :

### Scénario faible risque

- vibration proche de la médiane ;
- température moteur autour de 50 degrés ;
- courant modéré ;
- mode `normal` ou `idle` ;
- heures depuis maintenance raisonnables.

Le modèle devrait retourner une probabilité faible et classer la machine comme sans risque immédiat.

### Scénario risque élevé

- vibration élevée ;
- température moteur élevée ;
- courant électrique élevé ;
- régime moteur important ;
- mode `peak` ;
- maintenance ancienne.

Le modèle devrait retourner une probabilité forte et recommander une intervention ou une surveillance renforcée.

Les tests de démonstration doivent être réalisés dans Streamlit après génération des artefacts :

```bash
python scripts/run_pipeline.py
streamlit run src/dashboard/app.py
```

---

## 15. Gouvernance, responsabilité et limites

La solution doit être considérée comme un outil d'aide à la décision, pas comme un système autonome remplaçant l'expertise humaine.

### Qualité et traçabilité

Le pipeline est traçable : dataset brut, preprocessing, modèles, seuils et métriques sont séparés. Les fichiers de résultats sont exportés dans `outputs/results/`, ce qui facilite l'audit.

### Risques

- **Dérive des données** : les machines réelles peuvent évoluer dans le temps.
- **Biais dataset** : le dataset est simulé et peut ne pas couvrir tous les cas industriels.
- **Faux négatifs** : une panne non détectée peut coûter cher.
- **Faux positifs** : trop d'alertes peuvent créer de la fatigue opérationnelle.
- **Interprétabilité partielle** : SHAP aide à expliquer, mais ne prouve pas une causalité.

### Mesures recommandées en production

- suivi mensuel des métriques ;
- surveillance du taux de faux négatifs ;
- recalibrage périodique du seuil ;
- réentraînement avec nouvelles données ;
- validation humaine des alertes critiques ;
- journalisation des prédictions.

---

## 16. Limites et pistes d'amélioration

### Limites

- Dataset simulé et volume limité à environ 24 000 lignes.
- Pas de vraie série temporelle exploitée malgré la présence d'un timestamp.
- Une seule tâche prédictive traitée, alors que le dataset permettrait aussi la régression du RUL ou la classification du type de panne.
- API REST non implémentée.
- Dashboard local, sans déploiement cloud.
- Hyperparamètres optimisés de manière raisonnable mais non exhaustive.

### Améliorations possibles

1. Ajouter une API FastAPI avec `/predict`, `/health` et `/model-info`.
2. Déployer le dashboard sur Streamlit Community Cloud ou une VM.
3. Ajouter un monitoring de drift et de performance.
4. Tester des modèles temporels si l'ordre chronologique est exploitable.
5. Ajouter une tâche secondaire : estimation du `rul_hours` ou classification `failure_type`.
6. Optimiser les hyperparamètres avec `RandomizedSearchCV` ou Optuna.
7. Ajouter des tests automatisés sur le chargement des artefacts et une prédiction type.
8. Produire des captures d'écran définitives du dashboard pour la soutenance.

---

## 17. Conclusion

Ce projet démontre une démarche Data Science complète appliquée à la maintenance prédictive industrielle. A partir d'un dataset brut Kaggle, la solution réalise une analyse exploratoire, prépare les données, entraîne plusieurs modèles, compare leurs performances, interprète les prédictions et propose un dashboard décisionnel.

Le modèle final retenu est le Hist Gradient Boosting, car il représente le meilleur compromis entre performance, stabilité, coût de calcul, taille du modèle et facilité d'intégration. Même si Random Forest obtient le meilleur F1-score, son gain reste limité au regard de son coût d'inférence et de sa taille de sérialisation. Le projet montre également que le Deep Learning n'est pas systématiquement supérieur : le MLP obtient des résultats corrects, mais n'apporte pas de gain suffisant par rapport aux modèles d'arbres.

La valeur métier de la solution réside dans sa capacité à transformer des signaux capteurs en information actionnable : probabilité de panne, niveau de risque, variables explicatives et comparaison des modèles. La solution reste perfectible, notamment sur l'API, le déploiement et le monitoring, mais elle constitue un MVP solide et explicable.

---

## 18. Annexes

### Arborescence principale

```text
projet-maintenance-predictive/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── outputs/
│   ├── figures/
│   ├── models/
│   └── results/
├── reports/
├── scripts/
├── src/
│   ├── dashboard/
│   ├── models/
│   └── preprocessing/
├── README.md
└── requirements.txt
```

### Commandes principales

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_pipeline.py
streamlit run src/dashboard/app.py
```

### Fichiers de résultats générés

- `outputs/results/baseline_results.csv`
- `outputs/results/rebalancing_comparison.csv`
- `outputs/results/ml_models_comparison.csv`
- `outputs/results/mlp_results.csv`
- `outputs/results/threshold_tuning.csv`
- `outputs/results/final_model_comparison.csv`
- `outputs/results/final_test_metrics.csv`
- `outputs/results/feature_importance.csv`

### Dépendances principales

Les dépendances sont figées dans `requirements.txt` pour améliorer la reproductibilité.
