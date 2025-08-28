"""Scaling laws data collector for malware detection models.

This module collects comprehensive information about training runs to enable
scaling laws analysis, including model parameters, training metrics, and performance data.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TrainingRunData:
    """Data structure for a single training run."""

    # Run identification
    run_id: str
    timestamp: str
    run_name: str

    # Model architecture
    model_type: str
    layers: List[int]

    # Training configuration
    learning_rate: float
    weight_decay: float
    batch_size: int
    epochs: int
    time_limit_seconds: int

    # Dataset information
    feature_count: int
    train_samples: int
    val_samples: int
    test_samples: int

    # Hardware and timing
    device: str = "cuda"

    # Optional fields (with defaults)
    total_parameters: Optional[int] = None
    trainable_parameters: Optional[int] = None

    # Training metrics
    final_train_loss: Optional[float] = None
    final_val_loss: Optional[float] = None
    best_val_loss: Optional[float] = None
    epochs_trained: Optional[int] = None

    # Evaluation metrics
    test_loss: Optional[float] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    roc_auc: Optional[float] = None

    # Training curves (full history)
    train_losses: Optional[List[float]] = None
    val_losses: Optional[List[float]] = None
    train_accuracies: Optional[List[float]] = None
    val_accuracies: Optional[List[float]] = None

    # Additional timing
    actual_training_time: Optional[float] = None
    memory_usage_mb: Optional[float] = None


class ScalingLawsCollector:
    """Collects and manages training run data for scaling laws analysis."""

    def __init__(self, output_dir: Path):
        """Initialize the collector.

        Args:
            output_dir: Directory to save collected data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data storage
        self.runs_data: List[TrainingRunData] = []
        self.current_run: Optional[TrainingRunData] = None

        # File paths
        self.runs_file = self.output_dir / "scaling_laws_runs.json"
        self.summary_file = self.output_dir / "scaling_laws_summary.csv"

        # Load existing data if available
        self._load_existing_data()

    def _load_existing_data(self):
        """Load existing runs data from file."""
        if self.runs_file.exists():
            try:
                with open(self.runs_file, "r") as f:
                    data = json.load(f)
                    self.runs_data = [TrainingRunData(**run) for run in data]
                logger.info(f"Loaded {len(self.runs_data)} existing runs from {self.runs_file}")
            except Exception as e:
                logger.warning(f"Failed to load existing runs data: {e}")
                self.runs_data = []

    def start_run(self, run_name: str, **kwargs) -> str:
        """Start a new training run and return run ID.

        Args:
            run_name: Name identifier for the run
            **kwargs: Additional run parameters

        Returns:
            Unique run ID
        """
        run_id = f"{int(time.time())}_{run_name}"
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        self.current_run = TrainingRunData(
            run_id=run_id, timestamp=timestamp, run_name=run_name, **kwargs
        )

        logger.info(f"Started new training run: {run_id}")
        return run_id

    def update_run_metrics(self, **kwargs):
        """Update current run with additional metrics.

        Args:
            **kwargs: Metrics to update
        """
        if self.current_run is None:
            logger.warning("No active run to update")
            return

        for key, value in kwargs.items():
            if hasattr(self.current_run, key):
                setattr(self.current_run, key, value)
            else:
                logger.warning(f"Unknown metric key: {key}")

    def complete_run(self, training_history: Optional[Dict[str, List[float]]] = None):
        """Complete the current training run and save data.

        Args:
            training_history: Optional training history from the model
        """
        if self.current_run is None:
            logger.warning("No active run to complete")
            return

        # Add training history if provided
        if training_history:
            self.current_run.train_losses = training_history.get("train_losses")
            self.current_run.val_losses = training_history.get("val_losses")
            self.current_run.train_accuracies = training_history.get("train_accuracies")
            self.current_run.val_accuracies = training_history.get("val_accuracies")

        # Add to runs data
        self.runs_data.append(self.current_run)

        # Save data
        self._save_data()

        logger.info(f"Completed training run: {self.current_run.run_id}")
        self.current_run = None

    def _save_data(self):
        """Save runs data to files."""
        # Save detailed JSON data
        with open(self.runs_file, "w") as f:
            json.dump([asdict(run) for run in self.runs_data], f, indent=2)

        # Save summary CSV for easy analysis
        summary_data = []
        for run in self.runs_data:
            summary = {
                "run_id": run.run_id,
                "timestamp": run.timestamp,
                "run_name": run.run_name,
                "model_type": run.model_type,
                "total_parameters": run.total_parameters,
                "layers": str(run.layers),
                "learning_rate": run.learning_rate,
                "weight_decay": run.weight_decay,
                "batch_size": run.batch_size,
                "epochs": run.epochs,
                "time_limit_seconds": run.time_limit_seconds,
                "feature_count": run.feature_count,
                "train_samples": run.train_samples,
                "val_samples": run.val_samples,
                "test_samples": run.test_samples,
                "final_train_loss": run.final_train_loss,
                "final_val_loss": run.final_val_loss,
                "best_val_loss": run.best_val_loss,
                "epochs_trained": run.epochs_trained,
                "test_loss": run.test_loss,
                "accuracy": run.accuracy,
                "precision": run.precision,
                "recall": run.recall,
                "f1_score": run.f1_score,
                "roc_auc": run.roc_auc,
                "device": run.device,
                "actual_training_time": run.actual_training_time,
            }
            summary_data.append(summary)

        df = pd.DataFrame(summary_data)
        df.to_csv(self.summary_file, index=False)

        logger.info(f"Saved scaling laws data to {self.output_dir}")

    def get_scaling_data(self) -> pd.DataFrame:
        """Get scaling data as a pandas DataFrame.

        Returns:
            DataFrame with scaling data for analysis
        """
        if not self.runs_data:
            return pd.DataFrame()

        # Create scaling-specific columns
        scaling_data = []
        for run in self.runs_data:
            scaling_row = {
                "run_id": run.run_id,
                "run_name": run.run_name,
                "model_size": run.total_parameters or 0,
                "log_model_size": np.log10(run.total_parameters)
                if run.total_parameters and run.total_parameters > 0
                else 0,
                "dataset_size": (run.train_samples or 0) + (run.val_samples or 0),
                "log_dataset_size": np.log10((run.train_samples or 0) + (run.val_samples or 0))
                if ((run.train_samples or 0) + (run.val_samples or 0)) > 0
                else 0,
                "compute_budget": run.time_limit_seconds,
                "log_compute_budget": np.log10(run.time_limit_seconds)
                if run.time_limit_seconds > 0
                else 0,
                "final_val_loss": run.final_val_loss or 0,
                "best_val_loss": run.best_val_loss or 0,
                "final_test_loss": run.test_loss or 0,
                "test_accuracy": run.accuracy or 0,
                "test_f1": run.f1_score or 0,
                "test_auc": run.roc_auc or 0,
                "learning_rate": run.learning_rate,
                "weight_decay": run.weight_decay,
                "batch_size": run.batch_size,
                "epochs_trained": run.epochs_trained or 0,
                "layers": str(run.layers),
                "feature_count": run.feature_count,
            }
            scaling_data.append(scaling_row)

        return pd.DataFrame(scaling_data)

    def export_for_plotting(self, output_file: Optional[Path] = None) -> Path:
        """Export data in a format suitable for scaling laws plotting.

        Args:
            output_file: Optional output file path

        Returns:
            Path to exported file
        """
        if output_file is None:
            output_file = self.output_dir / "scaling_laws_plotting_data.json"

        scaling_df = self.get_scaling_data()

        # Convert to JSON format suitable for plotting
        plotting_data = {
            "runs": scaling_df.to_dict("records"),
            "metadata": {
                "total_runs": len(self.runs_data),
                "export_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model_types": list({run.model_type for run in self.runs_data}),
                "parameter_ranges": {
                    "min_parameters": min((run.total_parameters or 0) for run in self.runs_data)
                    if self.runs_data
                    else 0,
                    "max_parameters": max((run.total_parameters or 0) for run in self.runs_data)
                    if self.runs_data
                    else 0,
                },
                "dataset_ranges": {
                    "min_samples": min(
                        (run.train_samples or 0) + (run.val_samples or 0) for run in self.runs_data
                    )
                    if self.runs_data
                    else 0,
                    "max_samples": max(
                        (run.train_samples or 0) + (run.val_samples or 0) for run in self.runs_data
                    )
                    if self.runs_data
                    else 0,
                },
            },
        }

        with open(output_file, "w") as f:
            json.dump(plotting_data, f, indent=2)

        logger.info(f"Exported plotting data to {output_file}")
        return output_file

    def get_run_summary(self) -> Dict[str, Any]:
        """Get a summary of all collected runs.

        Returns:
            Dictionary with run summary statistics
        """
        if not self.runs_data:
            return {}

        return {
            "total_runs": len(self.runs_data),
            "model_types": list({run.model_type for run in self.runs_data}),
            "parameter_ranges": {
                "min": min((run.total_parameters or 0) for run in self.runs_data),
                "max": max((run.total_parameters or 0) for run in self.runs_data),
                "mean": sum((run.total_parameters or 0) for run in self.runs_data)
                / len(self.runs_data),
            },
            "performance_ranges": {
                "best_val_loss": min((run.best_val_loss or float("inf")) for run in self.runs_data),
                "worst_val_loss": max(
                    (run.best_val_loss or float("-inf")) for run in self.runs_data
                ),
                "best_accuracy": max((run.accuracy or 0) for run in self.runs_data),
                "worst_accuracy": min((run.accuracy or 1) for run in self.runs_data),
            },
            "recent_runs": [
                {
                    "run_id": run.run_id,
                    "run_name": run.run_name,
                    "timestamp": run.timestamp,
                    "total_parameters": run.total_parameters,
                    "best_val_loss": run.best_val_loss,
                    "test_accuracy": run.accuracy,
                }
                for run in sorted(self.runs_data, key=lambda x: x.timestamp, reverse=True)[:5]
            ],
        }


# Convenience function for easy integration
def get_scaling_collector(output_dir: Union[str, Path]) -> ScalingLawsCollector:
    """Get a scaling laws collector instance.

    Args:
        output_dir: Directory to save collected data

    Returns:
        ScalingLawsCollector instance
    """
    return ScalingLawsCollector(Path(output_dir))
