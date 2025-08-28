#!/usr/bin/env python3
"""Analyze scaling laws data collected from training runs.

This script loads the collected scaling laws data and creates various plots
to analyze how model performance scales with model size, dataset size, and compute budget.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set plotting style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def load_scaling_data(data_file: Path) -> Dict[str, Any]:
    """Load scaling laws data from file.

    Args:
        data_file: Path to the scaling laws data file

    Returns:
        Dictionary containing runs data and metadata
    """
    if not data_file.exists():
        raise FileNotFoundError(f"Scaling laws data file not found: {data_file}")

    with open(data_file, "r") as f:
        data = json.load(f)

    return data


def create_scaling_plots(data: Dict[str, Any], output_dir: Path):
    """Create scaling laws plots.

    Args:
        data: Scaling laws data dictionary
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_df = pd.DataFrame(data["runs"])

    if runs_df.empty:
        print("No runs data found for plotting")
        return

    print(f"Creating scaling plots for {len(runs_df)} runs...")

    # 1. Model Size vs Performance
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Scaling Laws Analysis: Model Size vs Performance", fontsize=16, fontweight="bold")

    # Model size vs validation loss
    axes[0, 0].scatter(runs_df["model_size"], runs_df["best_val_loss"], alpha=0.7, s=50)
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_xlabel("Model Parameters (log scale)")
    axes[0, 0].set_ylabel("Best Validation Loss")
    axes[0, 0].set_title("Model Size vs Validation Loss")
    axes[0, 0].grid(True, alpha=0.3)

    # Model size vs test accuracy
    axes[0, 1].scatter(runs_df["model_size"], runs_df["test_accuracy"], alpha=0.7, s=50)
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_xlabel("Model Parameters (log scale)")
    axes[0, 1].set_ylabel("Test Accuracy")
    axes[0, 1].set_title("Model Size vs Test Accuracy")
    axes[0, 1].grid(True, alpha=0.3)

    # Log model size vs validation loss
    axes[1, 0].scatter(runs_df["log_model_size"], runs_df["best_val_loss"], alpha=0.7, s=50)
    axes[1, 0].set_xlabel("Log10(Model Parameters)")
    axes[1, 0].set_ylabel("Best Validation Loss")
    axes[1, 0].set_title("Log Model Size vs Validation Loss")
    axes[1, 0].grid(True, alpha=0.3)

    # Log model size vs test accuracy
    axes[1, 1].scatter(runs_df["log_model_size"], runs_df["test_accuracy"], alpha=0.7, s=50)
    axes[1, 1].set_xlabel("Log10(Model Parameters)")
    axes[1, 1].set_ylabel("Test Accuracy")
    axes[1, 1].set_title("Log Model Size vs Test Accuracy")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "model_size_vs_performance.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Compute Budget vs Performance
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        "Scaling Laws Analysis: Compute Budget vs Performance", fontsize=16, fontweight="bold"
    )

    # Compute budget vs validation loss
    axes[0, 0].scatter(runs_df["compute_budget"], runs_df["best_val_loss"], alpha=0.7, s=50)
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_xlabel("Compute Budget (seconds, log scale)")
    axes[0, 0].set_ylabel("Best Validation Loss")
    axes[0, 0].set_title("Compute Budget vs Validation Loss")
    axes[0, 0].grid(True, alpha=0.3)

    # Compute budget vs test accuracy
    axes[0, 1].scatter(runs_df["compute_budget"], runs_df["test_accuracy"], alpha=0.7, s=50)
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_xlabel("Compute Budget (seconds, log scale)")
    axes[0, 1].set_ylabel("Test Accuracy")
    axes[0, 1].set_title("Compute Budget vs Test Accuracy")
    axes[0, 1].grid(True, alpha=0.3)

    # Log compute budget vs validation loss
    axes[1, 0].scatter(runs_df["log_compute_budget"], runs_df["best_val_loss"], alpha=0.7, s=50)
    axes[1, 0].set_xlabel("Log10(Compute Budget)")
    axes[1, 0].set_ylabel("Best Validation Loss")
    axes[1, 0].set_title("Log Compute Budget vs Validation Loss")
    axes[1, 0].grid(True, alpha=0.3)

    # Log compute budget vs test accuracy
    axes[1, 1].scatter(runs_df["log_compute_budget"], runs_df["test_accuracy"], alpha=0.7, s=50)
    axes[1, 1].set_xlabel("Log10(Compute Budget)")
    axes[1, 1].set_ylabel("Test Accuracy")
    axes[1, 1].set_title("Log Compute Budget vs Test Accuracy")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "compute_budget_vs_performance.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Dataset Size vs Performance
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        "Scaling Laws Analysis: Dataset Size vs Performance", fontsize=16, fontweight="bold"
    )

    # Dataset size vs validation loss
    axes[0, 0].scatter(runs_df["dataset_size"], runs_df["best_val_loss"], alpha=0.7, s=50)
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_xlabel("Dataset Size (samples, log scale)")
    axes[0, 0].set_ylabel("Best Validation Loss")
    axes[0, 0].set_title("Dataset Size vs Validation Loss")
    axes[0, 0].grid(True, alpha=0.3)

    # Dataset size vs test accuracy
    axes[0, 1].scatter(runs_df["dataset_size"], runs_df["test_accuracy"], alpha=0.7, s=50)
    axes[0, 1].set_xscale("log")
    axes[0, 0].set_xlabel("Dataset Size (samples, log scale)")
    axes[0, 1].set_ylabel("Test Accuracy")
    axes[0, 1].set_title("Dataset Size vs Test Accuracy")
    axes[0, 1].grid(True, alpha=0.3)

    # Log dataset size vs validation loss
    axes[1, 0].scatter(runs_df["log_dataset_size"], runs_df["best_val_loss"], alpha=0.7, s=50)
    axes[1, 0].set_xlabel("Log10(Dataset Size)")
    axes[1, 0].set_ylabel("Best Validation Loss")
    axes[1, 0].set_title("Log Dataset Size vs Validation Loss")
    axes[1, 0].grid(True, alpha=0.3)

    # Log dataset size vs test accuracy
    axes[1, 1].scatter(runs_df["log_dataset_size"], runs_df["test_accuracy"], alpha=0.7, s=50)
    axes[1, 1].set_xlabel("Log10(Dataset Size)")
    axes[1, 1].set_ylabel("Test Accuracy")
    axes[1, 1].set_title("Log Dataset Size vs Test Accuracy")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "dataset_size_vs_performance.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Combined scaling analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Combined Scaling Laws Analysis", fontsize=16, fontweight="bold")

    # 3D scatter: Model size vs Compute budget vs Performance
    scatter = axes[0, 0].scatter(
        runs_df["log_model_size"],
        runs_df["log_compute_budget"],
        c=runs_df["test_accuracy"],
        cmap="viridis",
        s=50,
        alpha=0.7,
    )
    axes[0, 0].set_xlabel("Log10(Model Parameters)")
    axes[0, 0].set_ylabel("Log10(Compute Budget)")
    axes[0, 0].set_title("Model Size vs Compute vs Accuracy")
    plt.colorbar(scatter, ax=axes[0, 0], label="Test Accuracy")
    axes[0, 0].grid(True, alpha=0.3)

    # Learning rate vs performance
    axes[0, 1].scatter(runs_df["learning_rate"], runs_df["test_accuracy"], alpha=0.7, s=50)
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_xlabel("Learning Rate (log scale)")
    axes[0, 1].set_ylabel("Test Accuracy")
    axes[0, 1].set_title("Learning Rate vs Test Accuracy")
    axes[0, 1].grid(True, alpha=0.3)

    # Weight decay vs performance
    axes[1, 0].scatter(runs_df["weight_decay"], runs_df["test_accuracy"], alpha=0.7, s=50)
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_xlabel("Weight Decay (log scale)")
    axes[1, 0].set_ylabel("Test Accuracy")
    axes[1, 0].set_title("Weight Decay vs Test Accuracy")
    axes[1, 0].grid(True, alpha=0.3)

    # Epochs trained vs performance
    axes[1, 1].scatter(runs_df["epochs_trained"], runs_df["test_accuracy"], alpha=0.7, s=50)
    axes[1, 1].set_xlabel("Epochs Trained")
    axes[1, 1].set_ylabel("Test Accuracy")
    axes[1, 1].set_title("Epochs Trained vs Test Accuracy")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "combined_scaling_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Scaling plots saved to {output_dir}")


def create_summary_report(data: Dict[str, Any], output_dir: Path):
    """Create a summary report of the scaling laws data.

    Args:
        data: Scaling laws data dictionary
        output_dir: Directory to save the report
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_df = pd.DataFrame(data["runs"])
    metadata = data.get("metadata", {})

    report_file = output_dir / "scaling_laws_summary_report.txt"

    with open(report_file, "w") as f:
        f.write("SCALING LAWS ANALYSIS SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Total Runs Analyzed: {metadata.get('total_runs', len(runs_df))}\n")
        f.write(f"Export Timestamp: {metadata.get('export_timestamp', 'N/A')}\n")
        f.write(f"Model Types: {', '.join(metadata.get('model_types', ['N/A']))}\n\n")

        if not runs_df.empty:
            f.write("PERFORMANCE STATISTICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Best Test Accuracy: {runs_df['test_accuracy'].max():.4f}\n")
            f.write(f"Worst Test Accuracy: {runs_df['test_accuracy'].min():.4f}\n")
            f.write(f"Mean Test Accuracy: {runs_df['test_accuracy'].mean():.4f}\n")
            f.write(f"Std Test Accuracy: {runs_df['test_accuracy'].std():.4f}\n\n")

            f.write(f"Best Validation Loss: {runs_df['best_val_loss'].min():.4f}\n")
            f.write(f"Worst Validation Loss: {runs_df['best_val_loss'].max():.4f}\n")
            f.write(f"Mean Validation Loss: {runs_df['best_val_loss'].mean():.4f}\n")
            f.write(f"Std Validation Loss: {runs_df['best_val_loss'].std():.4f}\n\n")

            f.write("MODEL SIZE STATISTICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Smallest Model: {runs_df['model_size'].min():,} parameters\n")
            f.write(f"Largest Model: {runs_df['model_size'].max():,} parameters\n")
            f.write(f"Mean Model Size: {runs_df['model_size'].mean():,.0f} parameters\n\n")

            f.write("COMPUTE BUDGET STATISTICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Shortest Training: {runs_df['compute_budget'].min():.0f} seconds\n")
            f.write(f"Longest Training: {runs_df['compute_budget'].max():.0f} seconds\n")
            f.write(f"Mean Training Time: {runs_df['compute_budget'].mean():.0f} seconds\n\n")

            f.write("TOP PERFORMING RUNS\n")
            f.write("-" * 30 + "\n")
            top_runs = runs_df.nlargest(5, "test_accuracy")[
                ["run_name", "test_accuracy", "model_size", "compute_budget"]
            ]
            for _, run in top_runs.iterrows():
                f.write(f"Run: {run['run_name']}\n")
                f.write(f"  Accuracy: {run['test_accuracy']:.4f}\n")
                f.write(f"  Model Size: {run['model_size']:,} parameters\n")
                f.write(f"  Compute Budget: {run['compute_budget']:.0f} seconds\n\n")

        f.write("SCALING LAWS INSIGHTS\n")
        f.write("-" * 30 + "\n")
        f.write("This report provides data for scaling laws analysis.\n")
        f.write("Key relationships to investigate:\n")
        f.write("1. Model size vs performance (validation loss, test accuracy)\n")
        f.write("2. Compute budget vs performance\n")
        f.write("3. Dataset size vs performance\n")
        f.write("4. Combined effects of multiple scaling factors\n")
        f.write("5. Optimal hyperparameter configurations\n")

    print(f"Summary report saved to {report_file}")


def main():
    """Main function to analyze scaling laws data."""
    parser = argparse.ArgumentParser(description="Analyze scaling laws data from training runs")
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path("reports/scaling_laws/scaling_laws_plotting_data.json"),
        help="Path to scaling laws data file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/scaling_laws/analysis"),
        help="Directory to save analysis plots and reports",
    )

    args = parser.parse_args()

    try:
        # Load data
        print(f"Loading scaling laws data from {args.data_file}...")
        data = load_scaling_data(args.data_file)

        # Create plots
        print("Creating scaling laws plots...")
        create_scaling_plots(data, args.output_dir)

        # Create summary report
        print("Creating summary report...")
        create_summary_report(data, args.output_dir)

        print("\n🎉 Scaling laws analysis completed!")
        print(f"📊 Plots and reports saved to: {args.output_dir}")
        print("📈 Data summary:")
        print(f"   - Total runs: {len(data.get('runs', []))}")
        print(
            f"   - Model types: {', '.join(data.get('metadata', {}).get('model_types', ['N/A']))}"
        )

    except Exception as e:
        print(f"❌ Error during scaling laws analysis: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
