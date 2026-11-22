"""
utils.py
Shared helper functions: plotting and metrics utilities used across
notebooks and the Streamlit app.
"""

import matplotlib.pyplot as plt
import pandas as pd


def plot_yield_vs_feature(df: pd.DataFrame, feature: str, target: str = "BHET_Yield_pct",
                           hue: str = "Catalyst", save_path: str = None):
    """Scatter plot of target vs a chosen numeric feature, colored by catalyst."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for catalyst, group in df.groupby(hue):
        ax.scatter(group[feature], group[target], label=catalyst, alpha=0.7)
    ax.set_xlabel(feature)
    ax.set_ylabel(target)
    ax.set_title(f"{target} vs {feature}")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_feature_importance(model, feature_names, save_path: str = None):
    """Bar plot of feature importances for a tree-based model."""
    importances = model.feature_importances_
    fig, ax = plt.subplots(figsize=(6, 4))
    sorted_idx = importances.argsort()
    ax.barh([feature_names[i] for i in sorted_idx], importances[sorted_idx])
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
