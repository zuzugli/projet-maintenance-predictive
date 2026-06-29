# Dashboard décisionnel

# Objectif : interface permettant la saisie d'un scénario,
# l'affichage de la prédiction, la comparaison des modèles, et l'importance
# des variables, avec des graphiques interactifs.

import os
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.project_config import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, RESULTS_DIR, resolve_raw_data_path

st.set_page_config(page_title="Maintenance Prédictive - Dashboard", page_icon="🔧", layout="wide")

NUMERIC_FEATURES = [
    "vibration_rms", "temperature_motor", "current_phase_avg",
    "pressure_level", "rpm", "hours_since_maintenance", "ambient_temp",
]

FEATURE_LABELS = {
    "vibration_rms": "Vibration (RMS)",
    "temperature_motor": "Température moteur (°C)",
    "current_phase_avg": "Courant électrique (A)",
    "pressure_level": "Pression",
    "rpm": "Vitesse de rotation (RPM)",
    "hours_since_maintenance": "Heures depuis maintenance",
    "ambient_temp": "Température ambiante (°C)",
}

MACHINE_DESCRIPTIONS = {
    "CNC": "Machine-outil à commande numérique (fraisage, tournage).",
    "Pump": "Pompe industrielle - transfert de fluides.",
    "Compressor": "Compresseur d'air ou de gaz.",
    "Robotic Arm": "Bras robotisé pour assemblage ou manutention.",
}

MODE_DESCRIPTIONS = {
    "idle": "Veille - machine au ralenti, faible sollicitation.",
    "normal": "Fonctionnement standard.",
    "peak": "Charge maximale - risque de surchauffe accru.",
}

SENSOR_HELP = {
    "vibration_rms": "Vibrations mécaniques (RMS) - valeur élevée = déséquilibre ou usure des roulements.",
    "temperature_motor": "Température du bobinage moteur - surchauffe précurseur de panne.",
    "current_phase_avg": "Intensité électrique moyenne - augmente en cas de surcharge.",
    "pressure_level": "Pression dans le circuit hydraulique ou pneumatique.",
    "rpm": "Tours par minute de l'arbre moteur.",
    "hours_since_maintenance": "Temps depuis la dernière maintenance préventive.",
    "ambient_temp": "Température ambiante - influe sur le refroidissement de la machine.",
}


@st.cache_data
def load_histgb_threshold():
    metrics_path = RESULTS_DIR / "final_test_metrics.csv"
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        row = df[df["model"] == "Hist Gradient Boosting"]
        if not row.empty:
            return float(row["seuil_retenu"].values[0])
    return 0.70


@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
    model = joblib.load(MODELS_DIR / "final_model.pkl")
    return preprocessor, model


@st.cache_data
def load_raw_data():
    return pd.read_csv(resolve_raw_data_path())


@st.cache_data
def load_comparison_data():
    comparison_path = RESULTS_DIR / "final_model_comparison.csv"
    metrics_path = RESULTS_DIR / "final_test_metrics.csv"
    if comparison_path.exists() and metrics_path.exists():
        comparison = pd.read_csv(comparison_path)
        metrics = pd.read_csv(metrics_path)
        df = comparison.merge(metrics[["model", "precision", "roc_auc", "pr_auc", "seuil_retenu"]], on="model", how="left")
        df["Modèle"] = df["model"].replace({"Hist Gradient Boosting": "Hist Gradient Boosting (retenu)"})
        df = df.rename(columns={
            "f1": "F1-score",
            "recall": "Recall",
            "precision": "Precision",
            "roc_auc": "ROC-AUC",
            "pr_auc": "PR-AUC",
            "predict_time_ms": "Temps prédiction (ms)",
            "model_size_kb": "Taille fichier (Ko)",
            "seuil_retenu": "Seuil retenu",
        })
        return df[[
            "Modèle", "F1-score", "Recall", "Precision", "ROC-AUC", "PR-AUC",
            "Temps prédiction (ms)", "Taille fichier (Ko)", "Seuil retenu",
        ]]

    return pd.DataFrame({
        "Modèle": ["Régression Logistique", "Random Forest", "Hist Gradient Boosting (retenu)", "MLP (Deep Learning)"],
        "F1-score": [0.7646, 0.8902, 0.9135, 0.8203],
        "Recall": [0.8624, 0.9284, 0.9340, 0.8848],
        "Precision": [0.6868, 0.8551, 0.8938, 0.7646],
        "ROC-AUC": [0.9589, 0.9937, 0.9960, 0.9813],
        "PR-AUC": [0.8379, 0.9656, 0.9794, 0.8976],
        "Temps prédiction (ms)": [0.64, 47.69, 3.45, 58.11],
        "Taille fichier (Ko)": [1.5, 38084.0, 343.8, 41.3],
        "Seuil retenu": [0.60, 0.55, 0.70, 0.75],
    })


@st.cache_data
def load_confusion_matrices():
    metrics_path = RESULTS_DIR / "final_test_metrics.csv"
    y_test_path = PROCESSED_DIR / "y_test.csv"
    if metrics_path.exists() and y_test_path.exists():
        metrics = pd.read_csv(metrics_path)
        y_test = pd.read_csv(y_test_path).squeeze()
        positives = int(y_test.sum())
        negatives = int((y_test == 0).sum())
        matrices = {}
        for _, row in metrics.iterrows():
            model_name = row["model"]
            display_name = "Hist Gradient Boosting (retenu)" if model_name == "Hist Gradient Boosting" else model_name
            fn = int(row["FN"])
            fp = int(row["FP"])
            matrices[display_name] = {
                "TN": negatives - fp,
                "FP": fp,
                "FN": fn,
                "TP": positives - fn,
                "seuil": row["seuil_retenu"],
            }
        return matrices

    # Valeurs calculées à chaque seuil optimal (maximise F1) sur le test set (4809 lignes, 712 positifs)
    return {
        "Régression Logistique":          {"TN": 3817, "FP": 280, "FN": 98,  "TP": 614, "seuil": 0.60},
        "Random Forest":                   {"TN": 3985, "FP": 112, "FN": 51,  "TP": 661, "seuil": 0.55},
        "Hist Gradient Boosting (retenu)": {"TN": 4018, "FP": 79,  "FN": 47,  "TP": 665, "seuil": 0.70},
        "MLP (Deep Learning)":             {"TN": 3903, "FP": 194, "FN": 82,  "TP": 630, "seuil": 0.75},
    }


@st.cache_data
def load_feature_importance():
    importance_path = RESULTS_DIR / "feature_importance.csv"
    if importance_path.exists():
        df = pd.read_csv(importance_path)
        feature_labels = {
            "num__rpm": "Vitesse de rotation (rpm)",
            "num__temperature_motor": "Température moteur",
            "num__current_phase_avg": "Courant électrique",
            "num__vibration_rms": "Vibration",
            "num__pressure_level": "Pression",
            "cat__operating_mode_peak": "Mode 'peak'",
            "cat__operating_mode_idle": "Mode 'idle'",
            "num__hours_since_maintenance": "Heures depuis maintenance",
            "num__ambient_temp": "Température ambiante",
        }
        df["Variable"] = df["feature"].map(lambda x: feature_labels.get(x, x.replace("num__", "").replace("cat__", "")))
        df["Importance"] = df["importance_mean"]
        return df[["Variable", "Importance"]].sort_values("Importance", ascending=False).head(10)

    return pd.DataFrame({
        "Variable": ["Vitesse de rotation (rpm)", "Température moteur", "Vibration",
                     "Courant électrique", "Heures depuis maintenance", "Pression",
                     "Mode 'idle'", "Mode 'peak'"],
        "Importance": [0.4036, 0.3139, 0.1729, 0.1636, 0.0804, 0.0570, 0.0187, 0.0173],
    })


def build_input_features(vibration, temperature, current, pressure, rpm,
                          hours_maintenance, ambient_temp, machine_type, operating_mode):
    return pd.DataFrame([{
        "machine_type": machine_type,
        "vibration_rms": vibration,
        "temperature_motor": temperature,
        "current_phase_avg": current,
        "pressure_level": pressure,
        "rpm": rpm,
        "operating_mode": operating_mode,
        "hours_since_maintenance": hours_maintenance,
        "ambient_temp": ambient_temp,
    }])


def main():
    threshold = load_histgb_threshold()

    try:
        preprocessor, model = load_artifacts()
    except FileNotFoundError as exc:
        st.error("Artefacts modèle introuvables. Lancez `python scripts/run_pipeline.py` depuis la racine du projet.")
        st.exception(exc)
        st.stop()

    st.title("🔧 Maintenance Prédictive Industrielle")
    st.caption("Dashboard décisionnel - Projet Data Science")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Prédiction en temps réel",
        "📊 Comparaison des modèles",
        "🔍 Importance des variables",
        "📈 Exploration des données",
    ])

    # ============================================================
    # ONGLET 1 : SAISIE D'UN SCÉNARIO + PRÉDICTION
    # ============================================================
    with tab1:
        st.header("Simuler un scénario machine")
        st.write("Renseignez les valeurs des capteurs pour estimer le risque de panne dans les 24h.")

        col1, col2, col3 = st.columns(3)
        with col1:
            machine_type = st.selectbox("Type de machine", ["CNC", "Pump", "Compressor", "Robotic Arm"])
            st.caption(MACHINE_DESCRIPTIONS[machine_type])
            operating_mode = st.selectbox("Mode opératoire", ["idle", "normal", "peak"])
            st.caption(MODE_DESCRIPTIONS[operating_mode])
            vibration = st.slider("Vibration (RMS)", 0.3, 10.0, 1.3, 0.1,
                                  help=SENSOR_HELP["vibration_rms"])
        with col2:
            temperature = st.slider("Température moteur (°C)", 28.0, 95.0, 50.0, 1.0,
                                    help=SENSOR_HELP["temperature_motor"])
            current = st.slider("Courant électrique moyen (A)", 2.0, 35.0, 6.4, 0.5,
                                help=SENSOR_HELP["current_phase_avg"])
            pressure = st.slider("Niveau de pression", 10.0, 207.0, 46.0, 1.0,
                                 help=SENSOR_HELP["pressure_level"])
        with col3:
            rpm = st.slider("Vitesse de rotation (RPM)", 124.0, 4100.0, 856.0, 10.0,
                            help=SENSOR_HELP["rpm"])
            hours_maintenance = st.slider("Heures depuis dernière maintenance", 0.0, 576.0, 122.0, 1.0,
                                          help=SENSOR_HELP["hours_since_maintenance"])
            ambient_temp = st.slider("Température ambiante (°C)", 8.0, 18.0, 13.0, 0.5,
                                     help=SENSOR_HELP["ambient_temp"])

        if st.button("🔍 Estimer le risque de panne", type="primary"):
            X_input = build_input_features(vibration, temperature, current, pressure, rpm,
                                           hours_maintenance, ambient_temp, machine_type, operating_mode)
            X_processed = preprocessor.transform(X_input)
            # On remet les noms de colonnes après transform (le modèle a été
            # entraîné avec un DataFrame nommé ; sans ça, sklearn émet un
            # warning inoffensif, mais propre à éviter)
            X_processed = pd.DataFrame(X_processed, columns=preprocessor.get_feature_names_out())
            proba = model.predict_proba(X_processed)[0, 1]

            st.divider()
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.metric("Probabilité de panne dans les 24h", f"{proba*100:.1f}%")
                if proba >= threshold:
                    st.error("⚠️ Risque élevé - intervention recommandée")
                elif proba >= 0.40:
                    st.warning("🟡 Risque modéré - surveillance conseillée")
                else:
                    st.success("✅ Risque faible")
            with col_b:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=proba * 100,
                    title={"text": "Risque de panne (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "darkred" if proba >= threshold else ("orange" if proba >= 0.4 else "green")},
                        "steps": [
                            {"range": [0, 40], "color": "#d4edda"},
                            {"range": [40, threshold * 100], "color": "#fff3cd"},
                            {"range": [threshold * 100, 100], "color": "#f8d7da"},
                        ],
                    },
                ))
                fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

            if proba >= threshold:
                st.warning(f"Le seuil de décision retenu ({threshold}) classe ce scénario comme **à risque de panne**.")
            else:
                st.success(f"Le seuil de décision retenu ({threshold}) classe ce scénario comme **sans risque immédiat**.")

    # ============================================================
    # ONGLET 2 : COMPARAISON DES MODÈLES + ÉCORESPONSABILITÉ
    # ============================================================
    with tab2:
        st.header("Comparaison des 4 modèles testés")
        st.write("Modèle retenu : **Hist Gradient Boosting** - meilleur compromis performance / coût de calcul / facilité de déploiement.")

        df_comp = load_comparison_data()
        st.dataframe(df_comp.style.format({
            "F1-score": "{:.4f}",
            "Recall": "{:.4f}",
            "Precision": "{:.4f}",
            "ROC-AUC": "{:.4f}",
            "PR-AUC": "{:.4f}",
            "Temps prédiction (ms)": "{:.2f}",
            "Taille fichier (Ko)": "{:.1f}",
        }, na_rep="N/A"), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig_perf = px.bar(
                df_comp, x="Modèle", y=["F1-score", "Recall", "Precision"],
                barmode="group", title="Performance des modèles",
            )
            st.plotly_chart(fig_perf, use_container_width=True)
        with col2:
            fig_cost = px.bar(
                df_comp, x="Modèle", y="Taille fichier (Ko)",
                title="Taille du fichier modèle (Ko, échelle log)", log_y=True,
            )
            st.plotly_chart(fig_cost, use_container_width=True)

        # --- Matrice de confusion ---
        st.divider()
        st.subheader("Matrice de confusion")
        st.write(
            "Visualise les **bonnes prédictions** (vert) et les **erreurs** (rouge) "
            "pour chaque modèle à son seuil de décision optimal."
        )

        cm_data = load_confusion_matrices()
        selected_cm = st.selectbox(
            "Sélectionner un modèle",
            options=list(cm_data.keys()),
            index=2,
            key="cm_model",
        )
        cm = cm_data[selected_cm]

        # Matrice 2×2 : couleur verte pour les cases correctes, rouge pour les erreurs
        fig_cm = go.Figure(go.Heatmap(
            z=[[1, 0], [0, 1]],
            x=["Prédit : Pas de panne", "Prédit : Panne"],
            y=["Réel : Pas de panne", "Réel : Panne"],
            colorscale=[[0, "#f8d7da"], [1, "#d4edda"]],
            text=[
                [f"TN = {cm['TN']}", f"FP = {cm['FP']}"],
                [f"FN = {cm['FN']}", f"TP = {cm['TP']}"],
            ],
            texttemplate="%{text}",
            textfont={"size": 18},
            showscale=False,
        ))
        fig_cm.update_layout(
            title=f"Matrice de confusion - {selected_cm} (seuil optimal = {cm['seuil']})",
            height=320,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        # --- Écoresponsabilité ---
        st.divider()
        st.subheader("Écoresponsabilité des modèles")
        st.write(
            "Le choix du modèle ne dépend pas uniquement de la performance prédictive. "
            "Dans un contexte industriel, **l'empreinte computationnelle** est un critère important : "
            "un modèle déployé effectue des milliers de prédictions quotidiennes, et son coût "
            "énergétique s'accumule dans le temps."
        )

        k1, k2, k3 = st.columns(3)
        histgb_row = df_comp[df_comp["Modèle"].str.contains("Hist Gradient")]
        rf_row = df_comp[df_comp["Modèle"] == "Random Forest"]
        if not histgb_row.empty and not rf_row.empty:
            gb_size = histgb_row["Taille fichier (Ko)"].values[0]
            gb_time = histgb_row["Temps prédiction (ms)"].values[0]
            rf_size = rf_row["Taille fichier (Ko)"].values[0]
            rf_time = rf_row["Temps prédiction (ms)"].values[0]
            size_pct = (1 - gb_size / rf_size) * 100
            time_pct = (1 - gb_time / rf_time) * 100
            k1.metric("Taille HistGB (retenu)", f"{gb_size:.0f} Ko", delta=f"-{size_pct:.0f} % vs Random Forest", delta_color="normal")
            k2.metric("Temps prédiction HistGB", f"{gb_time:.2f} ms", delta=f"-{time_pct:.0f} % vs Random Forest", delta_color="normal")
            k3.metric("Taille Random Forest", f"{rf_size:.0f} Ko", help=f"{rf_size/gb_size:.0f}x plus lourd que HistGB")
        else:
            k1.metric("Taille HistGB (retenu)", "344 Ko", delta="-99 % vs Random Forest", delta_color="normal")
            k2.metric("Temps prédiction HistGB", "3.45 ms", delta="-93 % vs Random Forest", delta_color="normal")
            k3.metric("Taille Random Forest", "38 084 Ko", help="111x plus lourd que HistGB")

        fig_eco = px.bar(
            df_comp.dropna(subset=["Temps prédiction (ms)"]),
            x="Modèle", y="Temps prédiction (ms)",
            color="Modèle",
            title="Temps de prédiction par modèle (ms) - moins c'est élevé, mieux c'est",
        )
        st.plotly_chart(fig_eco, use_container_width=True)

        if not histgb_row.empty and not rf_row.empty:
            eco_txt = (
                f"**Conclusion écoresponsabilité :** le modèle **Hist Gradient Boosting** est le plus vertueux - "
                f"il offre les meilleures performances opérationnelles tout en minimisant la taille du modèle "
                f"({gb_size:.0f} Ko contre {rf_size/1024:.1f} Mo pour le Random Forest) "
                f"et le temps de prédiction ({gb_time:.2f} ms contre {rf_time:.2f} ms). "
                f"En déploiement industriel, cela se traduit par une consommation énergétique réduite "
                f"pour chaque inférence et des besoins d'infrastructure moindres. "
                f"Le MLP (Deep Learning) présente un coût d'entraînement plus élevé "
                f"sans apporter un gain de performance suffisant pour le justifier sur ce jeu de données."
            )
        else:
            eco_txt = (
                "**Conclusion écoresponsabilité :** le modèle **Hist Gradient Boosting** est le plus vertueux - "
                "il offre les meilleures performances opérationnelles tout en minimisant la taille du modèle "
                "et le temps de prédiction. En déploiement industriel, cela se traduit par une consommation "
                "énergétique réduite pour chaque inférence et des besoins d'infrastructure moindres. "
                "Le MLP (Deep Learning) présente un coût d'entraînement plus élevé "
                "sans apporter un gain de performance suffisant pour le justifier sur ce jeu de données."
            )
        st.info(eco_txt)

        # --- Conclusion ---
        st.divider()
        st.subheader("Modèle retenu")
        if not histgb_row.empty and not rf_row.empty:
            gb_f1 = histgb_row["F1-score"].values[0]
            rf_f1 = rf_row["F1-score"].values[0]
            size_ratio = rf_size / gb_size
            time_ratio = rf_time / gb_time
            conclusion = (
                f"**Hist Gradient Boosting** a été sélectionné comme modèle final. "
                f"Il surpasse le Random Forest en F1 ({gb_f1:.3f} vs {rf_f1:.3f}) "
                f"tout en étant {size_ratio:.0f}x plus léger ({gb_size:.0f} Ko vs {rf_size:.0f} Ko) "
                f"et {time_ratio:.0f}x plus rapide en prédiction ({gb_time:.2f} ms vs {rf_time:.2f} ms). "
                f"Sa stabilité en validation croisée, sa facilité de déploiement et son interprétabilité native "
                f"en font le meilleur compromis pour un usage industriel."
            )
        else:
            conclusion = (
                "**Hist Gradient Boosting** a été sélectionné comme modèle final. "
                "Meilleur compromis performance (F1 = 0.914 vs 0.890 pour le Random Forest), "
                "légèreté (344 Ko vs 38 084 Ko) et vitesse de prédiction (3.45 ms vs 47.69 ms). "
                "Sa stabilité en validation croisée, sa facilité de déploiement et son interprétabilité native "
                "en font le meilleur compromis pour un usage industriel."
            )
        st.success(conclusion)

    # ============================================================
    # ONGLET 3 : IMPORTANCE DES VARIABLES + SHAP
    # ============================================================
    with tab3:
        st.header("Quelles variables influencent le plus les prédictions ?")
        st.write("Mesuré par Permutation Importance sur le modèle final (Hist Gradient Boosting).")

        df_imp = load_feature_importance()
        fig_imp = px.bar(
            df_imp.sort_values("Importance"),
            x="Importance", y="Variable",
            orientation="h",
            title="Importance des variables (Permutation Importance)",
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        st.info(
            "**Lecture métier :** le modèle se base principalement sur la vitesse de rotation, "
            "la température moteur, le courant électrique et la vibration pour anticiper une panne ; "
            "un profil cohérent avec ce qu'un technicien de maintenance surveillerait naturellement."
        )

        # --- SHAP ---
        st.divider()
        st.subheader("Analyse SHAP - explicabilité individuelle et globale")
        st.write(
            "La méthode SHAP (*SHapley Additive exPlanations*) complète la Permutation Importance : "
            "elle indique non seulement quelles variables comptent, mais aussi dans quel **sens** "
            "elles orientent la prédiction (vers le risque de panne ou vers la sécurité)."
        )

        shap_path = FIGURES_DIR / "06_shap_summary.png"
        if os.path.exists(shap_path):
            st.image(
                str(shap_path),
                caption="SHAP Summary Plot - rouge : la valeur élevée de la variable pousse vers une prédiction de panne ; bleu : elle pousse vers l'absence de panne.",
                use_container_width=True,
            )
        else:
            st.warning("Figure SHAP non trouvée. Lancez `src/models/interpretability.py` pour la générer.")

    # ============================================================
    # ONGLET 4 : EXPLORATION DES DONNÉES
    # ============================================================
    with tab4:
        st.header("Exploration des données")
        df = load_raw_data()
        df_labeled = df.copy()
        df_labeled["Statut"] = df_labeled["failure_within_24h"].map(
            {0: "Pas de panne", 1: "Panne dans 24h"}
        )

        # --- KPIs ---
        st.subheader("Indicateurs clés du dataset")
        n_total = len(df)
        n_failure = int(df["failure_within_24h"].sum())
        n_ok = n_total - n_failure
        failure_rate = n_failure / n_total * 100

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Enregistrements totaux", f"{n_total:,}")
        k2.metric("Cas avec panne (24h)", f"{n_failure:,}")
        k3.metric("Taux de panne", f"{failure_rate:.1f}%")
        k4.metric("Cas sans panne", f"{n_ok:,}")

        st.divider()

        # --- Distributions des capteurs ---
        st.subheader("Distributions des capteurs par classe")
        st.write(
            "Les histogrammes montrent la répartition de chaque capteur selon que la machine "
            "tombera en panne dans les 24h **(rouge)** ou non **(vert)**. "
            "Un écart visible entre les deux distributions indique un capteur prédictif."
        )

        selected_hist = st.selectbox(
            "Sélectionner un capteur",
            options=NUMERIC_FEATURES,
            format_func=lambda x: FEATURE_LABELS[x],
            key="hist_feature",
        )

        fig_dist = px.histogram(
            df_labeled,
            x=selected_hist,
            color="Statut",
            barmode="overlay",
            opacity=0.7,
            color_discrete_map={"Pas de panne": "#2ecc71", "Panne dans 24h": "#e74c3c"},
            labels={selected_hist: FEATURE_LABELS[selected_hist]},
            title=f"Distribution de « {FEATURE_LABELS[selected_hist]} » selon la classe",
            nbins=50,
        )
        fig_dist.update_layout(legend_title_text="Classe")
        st.plotly_chart(fig_dist, use_container_width=True)

        # --- Boxplots ---
        st.subheader("Boxplots : comparaison par classe")
        selected_box = st.selectbox(
            "Sélectionner un capteur",
            options=NUMERIC_FEATURES,
            format_func=lambda x: FEATURE_LABELS[x],
            key="box_feature",
        )

        fig_box = px.box(
            df_labeled,
            x="Statut",
            y=selected_box,
            color="Statut",
            color_discrete_map={"Pas de panne": "#2ecc71", "Panne dans 24h": "#e74c3c"},
            labels={selected_box: FEATURE_LABELS[selected_box]},
            title=f"Boxplot de « {FEATURE_LABELS[selected_box]} » par classe",
        )
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

        st.divider()

        # --- Heatmap de corrélations ---
        st.subheader("Matrice de corrélations entre les capteurs")
        st.write(
            "Une valeur proche de **+1** indique une forte relation positive, "
            "proche de **-1** une relation inverse, proche de **0** l'absence de relation linéaire."
        )

        corr_cols = NUMERIC_FEATURES + ["failure_within_24h"]
        corr_labels = {**FEATURE_LABELS, "failure_within_24h": "Panne 24h"}
        corr_matrix = df[corr_cols].corr()
        corr_matrix.index = [corr_labels.get(c, c) for c in corr_matrix.index]
        corr_matrix.columns = [corr_labels.get(c, c) for c in corr_matrix.columns]

        fig_corr = px.imshow(
            corr_matrix,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title="Heatmap de corrélations",
            text_auto=".2f",
        )
        fig_corr.update_layout(height=520)
        st.plotly_chart(fig_corr, use_container_width=True)

        st.info(
            "**Points clés :** la température moteur et la vibration sont les variables les plus corrélées "
            "avec le risque de panne (respectivement 0,39 et 0,26). "
            "On observe également une forte corrélation entre vibration, courant, pression et RPM (0,62–0,88), "
            "ce qui s'explique par les différents profils de machines présents dans le dataset."
        )


if __name__ == "__main__":
    main()
