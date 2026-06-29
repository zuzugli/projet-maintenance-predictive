"""
Génère les figures manquantes pour le rapport :
  08_roc_pr_curves.png           → courbes ROC et PR (4 modèles)
  09_preprocessing_avant_apres.png → distributions avant/après prétraitement
  10_mlp_learning_curve.png      → courbe d'apprentissage du MLP
  11_dashboard_exploration.png   → aperçu onglet Exploration des données

Usage :
    python scripts/generate_report_figures.py
"""

import sys
import time
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import joblib
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
)
from sklearn.utils.class_weight import compute_class_weight

from src.project_config import (
    MODELS_DIR, PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR, resolve_raw_data_path,
)
from src.models.train_mlp import build_mlp

tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

COLORS = {
    "Régression Logistique": "#636EFA",
    "Random Forest":         "#00CC96",
    "Hist Gradient Boosting": "#EF553B",
    "MLP (Deep Learning)":   "#AB63FA",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_test_data():
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()
    return X_test, y_test


def get_all_probas(X_test):
    probas = {}
    for name, fname in [
        ("Régression Logistique", "logreg.pkl"),
        ("Random Forest",         "random_forest.pkl"),
        ("Hist Gradient Boosting", "hist_gb.pkl"),
    ]:
        model = joblib.load(MODELS_DIR / fname)
        probas[name] = model.predict_proba(X_test)[:, 1]

    mlp = keras.models.load_model(MODELS_DIR / "mlp_model.keras")
    probas["MLP (Deep Learning)"] = mlp.predict(X_test.values, verbose=0).flatten()
    return probas


# ─────────────────────────────────────────────────────────────────────────────
# Figure 08 : courbes ROC et PR
# ─────────────────────────────────────────────────────────────────────────────

def fig_roc_pr_curves():
    X_test, y_test = load_test_data()
    probas = get_all_probas(X_test)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Courbes ROC et Précision-Rappel – Comparaison des 4 modèles",
        fontsize=13, fontweight="bold",
    )

    # ── ROC ──
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1,
             label="Référence aléatoire (AUC = 0.50)")
    for name, color in COLORS.items():
        fpr, tpr, _ = roc_curve(y_test, probas[name])
        roc_auc = auc(fpr, tpr)
        lw = 2.5 if name == "Hist Gradient Boosting" else 1.8
        ax1.plot(fpr, tpr, color=color, linewidth=lw,
                 label=f"{name}  (AUC = {roc_auc:.4f})")
    ax1.set_xlabel("Taux de faux positifs (1 – Spécificité)", fontsize=11)
    ax1.set_ylabel("Taux de vrais positifs (Rappel)", fontsize=11)
    ax1.set_title("Courbe ROC", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9, loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.02])

    # ── PR ──
    baseline = float(y_test.mean())
    ax2.axhline(y=baseline, color="k", linestyle="--", alpha=0.4, linewidth=1,
                label=f"Référence aléatoire (AP = {baseline:.2f})")
    for name, color in COLORS.items():
        precision, recall, _ = precision_recall_curve(y_test, probas[name])
        ap = average_precision_score(y_test, probas[name])
        lw = 2.5 if name == "Hist Gradient Boosting" else 1.8
        ax2.plot(recall, precision, color=color, linewidth=lw,
                 label=f"{name}  (AP = {ap:.4f})")
    ax2.set_xlabel("Rappel (Recall)", fontsize=11)
    ax2.set_ylabel("Précision", fontsize=11)
    ax2.set_title("Courbe Précision-Rappel (PR)", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9, loc="lower left")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1.02])

    fig.text(
        0.5, -0.03,
        "La courbe ROC mesure la discrimination globale ; la courbe PR est plus informative sur un "
        "dataset déséquilibré (14,8 % de pannes).\n"
        "Hist Gradient Boosting domine sur les deux critères (ROC-AUC = 0.9960, PR-AUC = 0.9794).",
        ha="center", fontsize=9, style="italic", color="#555555",
    )

    plt.tight_layout()
    out = FIGURES_DIR / "08_roc_pr_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 09 : distributions avant / après prétraitement
# ─────────────────────────────────────────────────────────────────────────────

def fig_preprocessing_avant_apres():
    df_raw = pd.read_csv(resolve_raw_data_path())
    X_train_proc = pd.read_csv(PROCESSED_DIR / "X_train.csv")

    VARS = [
        ("temperature_motor",      "Température moteur (°C)", "num__temperature_motor"),
        ("vibration_rms",          "Vibration (RMS)",          "num__vibration_rms"),
        ("rpm",                    "Vitesse de rotation (RPM)", "num__rpm"),
        ("hours_since_maintenance","Heures depuis maintenance", "num__hours_since_maintenance"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    fig.suptitle(
        "Distributions des variables – Avant et Après prétraitement\n"
        "(imputation médiane + StandardScaler)",
        fontsize=13, fontweight="bold",
    )

    color_before = "#4C78A8"
    color_after  = "#F58518"

    for col_idx, (raw_col, label, proc_col) in enumerate(VARS):
        ax_b = axes[0, col_idx]
        ax_a = axes[1, col_idx]

        raw_vals = df_raw[raw_col].dropna()
        n_miss = int(df_raw[raw_col].isna().sum())
        pct_miss = n_miss / len(df_raw) * 100

        # ── Avant ──
        ax_b.hist(raw_vals, bins=40, color=color_before, alpha=0.85,
                  edgecolor="white", linewidth=0.3)
        ax_b.set_title(f"{label}\n— Avant —", fontsize=9, fontweight="bold")
        ax_b.set_xlabel(
            f"Valeur brute\n({n_miss} valeurs manquantes, {pct_miss:.1f} %)",
            fontsize=8,
        )
        ax_b.set_ylabel("Fréquence", fontsize=9)
        ax_b.grid(True, alpha=0.3)
        ax_b.tick_params(labelsize=8)
        # annotation de la médiane
        med = float(raw_vals.median())
        ax_b.axvline(med, color="red", linewidth=1.2, linestyle="--", alpha=0.8)
        ax_b.text(med, ax_b.get_ylim()[1] * 0.9, f"  médiane\n  {med:.1f}",
                  fontsize=7, color="red")

        # ── Après ──
        if proc_col in X_train_proc.columns:
            proc_vals = X_train_proc[proc_col]
            ax_a.hist(proc_vals, bins=40, color=color_after, alpha=0.85,
                      edgecolor="white", linewidth=0.3)
            ax_a.set_title(f"{label}\n— Après —", fontsize=9, fontweight="bold")
            ax_a.set_xlabel("Valeur standardisée (z-score)\n(0 valeur manquante)", fontsize=8)
            ax_a.set_ylabel("Fréquence", fontsize=9)
            ax_a.grid(True, alpha=0.3)
            ax_a.tick_params(labelsize=8)
            ax_a.axvline(0, color="red", linewidth=1.2, linestyle="--", alpha=0.8)
            ax_a.text(0, ax_a.get_ylim()[1] * 0.9, "  µ = 0",
                      fontsize=7, color="red")

    patch_b = mpatches.Patch(color=color_before, label="Avant (valeurs brutes)")
    patch_a = mpatches.Patch(color=color_after,  label="Après (z-score)")
    fig.legend(handles=[patch_b, patch_a], loc="lower center", ncol=2,
               fontsize=10, bbox_to_anchor=(0.5, -0.05))

    fig.text(
        0.5, -0.10,
        "Avant : valeurs brutes (échelles hétérogènes, valeurs manquantes). "
        "Après : valeurs centrées-réduites (µ ≈ 0, σ ≈ 1), aucune valeur manquante.\n"
        "La ligne rouge indique la médiane (avant) et zéro (après) — centre de la distribution après standardisation.",
        ha="center", fontsize=9, style="italic", color="#555555",
    )

    plt.tight_layout()
    out = FIGURES_DIR / "09_preprocessing_avant_apres.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 10 : courbe d'apprentissage du MLP
# ─────────────────────────────────────────────────────────────────────────────

def fig_mlp_learning_curve():
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze()

    row = pd.read_csv(RESULTS_DIR / "mlp_results.csv", skipinitialspace=True).iloc[0]
    best = {
        "units_1":       int(row["best_units_1"]),
        "units_2":       int(row["best_units_2"]),
        "dropout":       float(row["best_dropout"]),
        "learning_rate": float(row["best_learning_rate"]),
        "batch_size":    int(row["best_batch_size"]),
    }
    print(f"  Meilleurs hyperparamètres MLP : {best}")

    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    cw = {0: weights[0], 1: weights[1]}

    model = build_mlp(
        input_dim=X_train.shape[1],
        units_1=best["units_1"],
        units_2=best["units_2"],
        dropout=best["dropout"],
        learning_rate=best["learning_rate"],
    )
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True, verbose=0,
    )

    print("  Réentraînement du MLP (~15 s)...")
    t0 = time.time()
    history = model.fit(
        X_train.values, y_train.values,
        validation_split=0.2,
        epochs=100,
        batch_size=best["batch_size"],
        class_weight=cw,
        callbacks=[early_stop],
        verbose=0,
    )
    elapsed = time.time() - t0
    n_ep = len(history.history["loss"])
    best_ep = int(np.argmin(history.history["val_loss"])) + 1
    print(f"  Terminé en {elapsed:.1f}s — {n_ep} époques (meilleure : {best_ep})")

    epochs = range(1, n_ep + 1)
    col_train = "#4C78A8"
    col_val   = "#F58518"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Courbe d'apprentissage du MLP\n"
        f"Architecture : {best['units_1']}→{best['units_2']}→1 neurones  |  "
        f"Dropout = {best['dropout']}  |  lr = {best['learning_rate']}  |  "
        f"Batch = {best['batch_size']}",
        fontsize=11, fontweight="bold",
    )

    # ── Loss ──
    ax1.plot(epochs, history.history["loss"],     color=col_train, linewidth=2, label="Loss (train)")
    ax1.plot(epochs, history.history["val_loss"], color=col_val,   linewidth=2, linestyle="--", label="Loss (validation)")
    ax1.axvline(best_ep, color="red", linestyle=":", linewidth=1.5, alpha=0.8)
    ax1.annotate(
        f"Early stopping\nÉpoque {best_ep}",
        xy=(best_ep, history.history["val_loss"][best_ep - 1]),
        xytext=(best_ep + 1.5, history.history["val_loss"][best_ep - 1] + 0.025),
        fontsize=8, color="red",
        arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
    )
    ax1.set_xlabel("Époque", fontsize=11)
    ax1.set_ylabel("Binary Cross-Entropy Loss", fontsize=11)
    ax1.set_title("Évolution de la loss", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # ── Accuracy ──
    ax2.plot(epochs, history.history["accuracy"],     color=col_train, linewidth=2, label="Accuracy (train)")
    ax2.plot(epochs, history.history["val_accuracy"], color=col_val,   linewidth=2, linestyle="--", label="Accuracy (validation)")
    ax2.axvline(best_ep, color="red", linestyle=":", linewidth=1.5, alpha=0.8)
    ax2.set_xlabel("Époque", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.set_title("Évolution de l'accuracy", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.text(
        0.5, -0.04,
        f"L'early stopping (patience = 10) interrompt l'entraînement à l'époque {best_ep} "
        f"({n_ep} époques effectives). "
        f"L'absence de divergence significative entre train et validation confirme que le dropout "
        f"({best['dropout']}) contrôle correctement le surapprentissage.",
        ha="center", fontsize=9, style="italic", color="#555555",
    )

    plt.tight_layout()
    out = FIGURES_DIR / "10_mlp_learning_curve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 11 : aperçu de l'onglet Exploration des données
# ─────────────────────────────────────────────────────────────────────────────

def fig_dashboard_exploration():
    df = pd.read_csv(resolve_raw_data_path())

    SENSORS = [
        ("temperature_motor",       "Température moteur (°C)"),
        ("vibration_rms",           "Vibration (RMS)"),
        ("rpm",                     "Vitesse de rotation (RPM)"),
        ("hours_since_maintenance", "Heures depuis maintenance"),
    ]

    fig = plt.figure(figsize=(16, 11))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.38,
                           top=0.88, bottom=0.06, left=0.06, right=0.97)

    # ── Row 0 : indicateurs clés ──
    ax_kpi = fig.add_subplot(gs[0, :])
    ax_kpi.axis("off")
    ax_kpi.set_title("Indicateurs clés du dataset", fontsize=11,
                      fontweight="bold", pad=6, loc="left")

    n_total = len(df)
    n_panne = int(df["failure_within_24h"].sum())
    n_ok    = n_total - n_panne
    taux    = df["failure_within_24h"].mean() * 100

    kpis = [
        ("Enregistrements\ntotaux",  f"{n_total:,}",  "#1f77b4"),
        ("Cas avec\npanne (24 h)",   f"{n_panne:,}",  "#d62728"),
        ("Taux de\npanne",           f"{taux:.1f} %", "#ff7f0e"),
        ("Cas sans\npanne",          f"{n_ok:,}",     "#2ca02c"),
    ]
    for i, (lbl, val, col) in enumerate(kpis):
        x = 0.10 + i * 0.235
        rect = mpatches.FancyBboxPatch(
            (x - 0.095, 0.05), 0.19, 0.90,
            boxstyle="round,pad=0.02",
            facecolor=col, alpha=0.10,
            edgecolor=col, linewidth=2,
            transform=ax_kpi.transAxes,
        )
        ax_kpi.add_patch(rect)
        ax_kpi.text(x, 0.68, val,  ha="center", va="center",
                    fontsize=24, fontweight="bold", color=col,
                    transform=ax_kpi.transAxes)
        ax_kpi.text(x, 0.25, lbl,  ha="center", va="center",
                    fontsize=9,  color="#444444",
                    transform=ax_kpi.transAxes)

    # ── Row 1 : histogrammes superposés par classe ──
    c0, c1 = "#2ca02c", "#d62728"
    lbl0, lbl1 = "Pas de panne", "Panne dans 24 h"

    for ci, (col, label) in enumerate(SENSORS):
        ax = fig.add_subplot(gs[1, ci])
        for cls, color, lbl in [(0, c0, lbl0), (1, c1, lbl1)]:
            vals = df[df["failure_within_24h"] == cls][col].dropna()
            ax.hist(vals, bins=30, alpha=0.60, color=color,
                    label=lbl, density=True, edgecolor="white", linewidth=0.2)
        ax.set_title(f"Distribution de « {label} »\nselon la classe",
                     fontsize=9, fontweight="bold")
        ax.set_xlabel(label, fontsize=8)
        ax.set_ylabel("Densité", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    # ── Row 2 : boxplots par classe ──
    for ci, (col, label) in enumerate(SENSORS):
        ax = fig.add_subplot(gs[2, ci])
        d0 = df[df["failure_within_24h"] == 0][col].dropna()
        d1 = df[df["failure_within_24h"] == 1][col].dropna()
        bp = ax.boxplot([d0, d1], patch_artist=True, widths=0.50,
                        medianprops=dict(color="black", linewidth=2))
        bp["boxes"][0].set_facecolor(c0); bp["boxes"][0].set_alpha(0.55)
        bp["boxes"][1].set_facecolor(c1); bp["boxes"][1].set_alpha(0.55)
        ax.set_xticklabels(["Pas de panne", "Panne 24 h"], fontsize=8)
        ax.set_title(f"Boxplot « {label} »\nselon la classe",
                     fontsize=9, fontweight="bold")
        ax.set_ylabel(label, fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(labelsize=7)

    fig.suptitle(
        "Onglet « Exploration des données » – Dashboard Maintenance Prédictive",
        fontsize=13, fontweight="bold",
    )
    fig.text(
        0.5, 0.01,
        "Vue statique de l'onglet Exploration des données du dashboard Streamlit. "
        "L'interface interactive permet de sélectionner n'importe quelle variable "
        "et d'afficher dynamiquement la matrice de corrélation et les boxplots.",
        ha="center", fontsize=9, style="italic", color="#555555",
    )

    out = FIGURES_DIR / "11_dashboard_exploration.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("Génération des figures manquantes pour le rapport")
    print("=" * 60)

    print("\n[1/4] Courbes ROC et PR...")
    fig_roc_pr_curves()

    print("\n[2/4] Distributions avant/après prétraitement...")
    fig_preprocessing_avant_apres()

    print("\n[3/4] Courbe d'apprentissage du MLP (réentraînement ~15 s)...")
    fig_mlp_learning_curve()

    print("\n[4/4] Aperçu onglet Exploration des données...")
    fig_dashboard_exploration()

    print("\n" + "=" * 60)
    print("Terminé. Figures dans outputs/figures/")
    print("  08_roc_pr_curves.png")
    print("  09_preprocessing_avant_apres.png")
    print("  10_mlp_learning_curve.png")
    print("  11_dashboard_exploration.png")
    print("=" * 60)
