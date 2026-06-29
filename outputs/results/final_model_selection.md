# Choix du modèle final

Modèle retenu : Hist Gradient Boosting.

Justification : après optimisation des hyperparamètres, Hist Gradient Boosting obtient le meilleur compromis global : meilleur F1/Recall, faible nombre de faux négatifs, coût de prédiction inférieur au Random Forest, taille de fichier réduite et intégration Streamlit simple. Les seuils de décision utilisés pour l'évaluation finale ont été sélectionnés sur une validation interne avec les mêmes hyperparamètres que les modèles finaux, puis appliqués une seule fois au test set.
