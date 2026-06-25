# Dashboard décisionnel 

# Objectif : interface permettant la saisie d'un scénario,
# l'affichage de la prédiction, la comparaison des modèles, et l'importance
# des variables, avec des graphiques interactifs.

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Maintenance Prédictive - Dashboard", page_icon="🔧", layout="wide")

MODELS_DIR = "outputs/models"
FIGURES_DIR = "outputs/figures"


@st.cache_resource
def load_artifacts():
    # Charge le préprocesseur et le modèle final une seule fois (mis en cache
    # par Streamlit pour ne pas recharger à chaque interaction utilisateur).
    preprocessor = joblib.load(f"{MODELS_DIR}/preprocessor.pkl")
    model = joblib.load(f"{MODELS_DIR}/final_model.pkl")
    return preprocessor, model


@st.cache_data
def load_comparison_data():
    #Recharge les résultats de comparaison des 4 modèles, pour l'onglet
    # "Comparaison des modèles". Valeurs reprises directement des résultats
    # obtenus en Phase D, pas recalculées ici pour garder le dashboard rapide à l'usage
    return pd.DataFrame({
        "Modèle": ["Régression Logistique", "Random Forest", "Hist Gradient Boosting (retenu)", "MLP (Deep Learning)"],
        "F1-score": [0.7658, 0.8625, 0.8618, 0.8208],
        "Recall": [0.8062, 0.8989, 0.8890, 0.8427],
        "Precision": [0.7294, 0.8290, 0.8362, 0.8000],
        "Temps prédiction (ms)": [0.73, 14.39, 2.12, np.nan],
        "Taille fichier (Ko)": [1.5, 8531.5, 213.4, 41.3],
    })


@st.cache_data
def load_feature_importance():
    # Recharge l'importance des variables
    return pd.DataFrame({
        "Variable": ["Vitesse de rotation (rpm)", "Température moteur", "Courant électrique",
                     "Vibration", "Pression", "Mode 'peak'", "Mode 'idle'",
                     "Heures depuis maintenance"],
        "Importance": [0.4110, 0.2622, 0.2104, 0.1930, 0.0715, 0.0421, 0.0147, 0.0114],
    })


def build_input_features(vibration, temperature, current, pressure, rpm,
                           hours_maintenance, ambient_temp, machine_type, operating_mode):
    # Construit un DataFrame d'une ligne avec les colonnes brutes exactement 
    # dans le même format que celui attendu par le preprocessor.pkl 
    # c'est lui qui se charge ensuite de l'imputation/encodage/scaling
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
    preprocessor, model = load_artifacts()

    st.title("🔧 Maintenance Prédictive Industrielle")
    st.caption("Dashboard décisionnel - Projet Data Science")

    tab1, tab2, tab3 = st.tabs(["🎯 Prédiction en temps réel", "📊 Comparaison des modèles", "🔍 Importance des variables"])

    # ============================================================
    # ONGLET 1 : SAISIE D'UN SCÉNARIO + PRÉDICTION
    # ============================================================
    with tab1:
        st.header("Simuler un scénario machine")
        st.write("Renseignez les valeurs des capteurs pour estimer le risque de panne dans les 24h.")

        col1, col2, col3 = st.columns(3)
        with col1:
            machine_type = st.selectbox("Type de machine", ["CNC", "Pump", "Compressor", "Robotic Arm"])
            operating_mode = st.selectbox("Mode opératoire", ["idle", "normal", "peak"])
            vibration = st.slider("Vibration (RMS)", 0.3, 10.0, 1.3, 0.1)
        with col2:
            temperature = st.slider("Température moteur (°C)", 28.0, 95.0, 50.0, 1.0)
            current = st.slider("Courant électrique moyen (A)", 2.0, 35.0, 6.4, 0.5)
            pressure = st.slider("Niveau de pression", 10.0, 207.0, 46.0, 1.0)
        with col3:
            rpm = st.slider("Vitesse de rotation (RPM)", 124.0, 4100.0, 856.0, 10.0)
            hours_maintenance = st.slider("Heures depuis dernière maintenance", 0.0, 576.0, 122.0, 1.0)
            ambient_temp = st.slider("Température ambiante (°C)", 8.0, 18.0, 13.0, 0.5)

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
                if proba >= 0.75:
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
                    gauge={"axis": {"range": [0, 100]},
                           "bar": {"color": "darkred" if proba >= 0.75 else ("orange" if proba >= 0.4 else "green")},
                           "steps": [{"range": [0, 40], "color": "#d4edda"},
                                     {"range": [40, 75], "color": "#fff3cd"},
                                     {"range": [75, 100], "color": "#f8d7da"}]},
                ))
                fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

            # Seuil retenu : 0.75 pour Hist Gradient Boosting (meilleur F1)
            if proba >= 0.75:
                st.warning("Le seuil de décision retenu (0.75) classe ce scénario comme **à risque de panne**.")
            else:
                st.success("Le seuil de décision retenu (0.75) classe ce scénario comme **sans risque immédiat**.")

    # ============================================================
    # ONGLET 2 : COMPARAISON DES MODÈLES
    # ============================================================
    with tab2:
        st.header("Comparaison des 4 modèles testés")
        st.write("Modèle retenu : **Hist Gradient Boosting** - meilleur compromis performance / coût de calcul / facilité de déploiement.")

        df_comp = load_comparison_data()
        st.dataframe(df_comp.style.format({
            "F1-score": "{:.4f}", "Recall": "{:.4f}", "Precision": "{:.4f}",
            "Temps prédiction (ms)": "{:.2f}", "Taille fichier (Ko)": "{:.1f}",
        }), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig_perf = px.bar(df_comp, x="Modèle", y=["F1-score", "Recall", "Precision"],
                               barmode="group", title="Performance des modèles")
            st.plotly_chart(fig_perf, use_container_width=True)
        with col2:
            fig_cost = px.bar(df_comp, x="Modèle", y="Taille fichier (Ko)",
                               title="Taille du fichier modèle (Ko, échelle log)", log_y=True)
            st.plotly_chart(fig_cost, use_container_width=True)

    # ============================================================
    # ONGLET 3 : IMPORTANCE DES VARIABLES 
    # ============================================================
    with tab3:
        st.header("Quelles variables influencent le plus les prédictions ?")
        st.write("Mesuré par Permutation Importance sur le modèle final (Hist Gradient Boosting).")

        df_imp = load_feature_importance()
        fig_imp = px.bar(df_imp.sort_values("Importance"), x="Importance", y="Variable",
                          orientation="h", title="Importance des variables")
        st.plotly_chart(fig_imp, use_container_width=True)

        st.info("""
        **Lecture métier :** le modèle se base principalement sur la vitesse de rotation, 
        la température moteur, le courant électrique et la vibration pour anticiper une panne ;
        un profil cohérent avec ce qu'un technicien de maintenance surveillerait naturellement. 
        """)


if __name__ == "__main__":
    main()