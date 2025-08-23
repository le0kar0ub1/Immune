"""Visualization utilities for malware detection results."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()

# Set style for plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def plot_training_history(
    history: Dict[str, List[float]], save_path: Optional[Path] = None, show_plot: bool = True
) -> None:
    """Plot training and validation loss curves.

    Args:
        history: Dictionary containing 'train_losses' and 'val_losses'
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    if "train_losses" not in history or "val_losses" not in history:
        console.print("[red]Error: History must contain 'train_losses' and 'val_losses'[/red]")
        return

    epochs = range(1, len(history["train_losses"]) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_losses"], "b-", label="Training Loss", linewidth=2)
    plt.plot(epochs, history["val_losses"], "r-", label="Validation Loss", linewidth=2)

    plt.title("Training and Validation Loss", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        console.print(f"[green]Plot saved to {save_path}[/green]")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, save_path: Optional[Path] = None, show_plot: bool = True
) -> None:
    """Plot confusion matrix for classification results.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benign", "Malware"],
        yticklabels=["Benign", "Malware"],
    )

    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        console.print(f"[green]Confusion matrix saved to {save_path}[/green]")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_feature_importance(
    feature_names: List[str],
    importance_scores: np.ndarray,
    top_n: int = 20,
    save_path: Optional[Path] = None,
    show_plot: bool = True,
) -> None:
    """Plot feature importance scores.

    Args:
        feature_names: List of feature names
        importance_scores: Array of importance scores
        top_n: Number of top features to display
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    if len(feature_names) != len(importance_scores):
        console.print("[red]Error: Feature names and importance scores must have same length[/red]")
        return

    # Sort features by importance
    indices = np.argsort(importance_scores)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_scores = importance_scores[indices]

    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(top_features))

    plt.barh(y_pos, top_scores, color="skyblue", edgecolor="navy", alpha=0.7)
    plt.yticks(y_pos, top_features)
    plt.xlabel("Importance Score", fontsize=12)
    plt.title(f"Top {top_n} Feature Importance Scores", fontsize=14, fontweight="bold")
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis="x")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        console.print(f"[green]Feature importance plot saved to {save_path}[/green]")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: Optional[Path] = None,
    show_plot: bool = True,
) -> None:
    """Plot ROC curve for classification results.

    Args:
        y_true: True labels
        y_scores: Prediction scores/probabilities
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    from sklearn.metrics import auc, roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        console.print(f"[green]ROC curve saved to {save_path}[/green]")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: Optional[Path] = None,
    show_plot: bool = True,
) -> None:
    """Plot precision-recall curve for classification results.

    Args:
        y_true: True labels
        y_scores: Prediction scores/probabilities
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    from sklearn.metrics import average_precision_score, precision_recall_curve

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(
        recall, precision, color="darkgreen", lw=2, label=f"PR curve (AP = {avg_precision:.2f})"
    )

    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        console.print(f"[green]Precision-recall curve saved to {save_path}[/green]")

    if show_plot:
        plt.show()
    else:
        plt.close()


def create_training_report(
    history: Dict[str, List[float]], metrics: Dict[str, float], save_dir: Path
) -> None:
    """Create a comprehensive training report with multiple plots.

    Args:
        history: Training history dictionary
        metrics: Dictionary of evaluation metrics
        save_dir: Directory to save the report plots
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold blue]📊 Creating training report...[/bold blue]")

    # Plot training history
    plot_training_history(history, save_path=save_dir / "training_history.png", show_plot=False)

    # Save metrics to file
    metrics_file = save_dir / "metrics.txt"
    with open(metrics_file, "w") as f:
        f.write("Training Report\n")
        f.write("=" * 50 + "\n\n")
        for metric_name, value in metrics.items():
            f.write(f"{metric_name}: {value:.4f}\n")

    console.print(f"[green]✅ Training report saved to {save_dir}[/green]")
