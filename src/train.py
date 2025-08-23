#!/usr/bin/env python3
"""Training script for the multi-headed malware detection model."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from rich.console import Console
from rich.logging import RichHandler
from torch.utils.data import DataLoader, random_split

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.feature_extractor import MultiHeadFeatureExtractor
from src.models.malware_detector import (
    MalwareDataset,
    MultiHeadMalwareDetector,
    train_multi_head_model,
)
from src.utils.visualization import create_training_report

console = Console()


def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def load_dataset(
    data_dir: Path, feature_extractor: MultiHeadFeatureExtractor, train_split: float = 0.8
) -> Tuple[DataLoader, DataLoader]:
    """Load and prepare dataset for training.

    Args:
        data_dir: Directory containing malware and benign samples
        feature_extractor: Feature extractor instance
        train_split: Fraction of data to use for training

    Returns:
        Tuple of (train_loader, val_loader)
    """
    console.print("[yellow]📁 Loading dataset...[/yellow]")

    # Expected directory structure:
    # data_dir/
    # ├── malware/
    # │   ├── sample1.exe
    # │   ├── sample2.exe
    # │   └── ...
    # └── benign/
    #     ├── sample1.exe
    #     ├── sample2.exe
    #     └── ...

    malware_dir = data_dir / "malware"
    benign_dir = data_dir / "benign"

    if not malware_dir.exists() or not benign_dir.exists():
        raise ValueError(f"Expected directory structure:\n{data_dir}/malware/\n{data_dir}/benign/")

    # Collect file paths and labels
    malware_files = list(malware_dir.glob("*.exe")) + list(malware_dir.glob("*.dll"))
    benign_files = list(benign_dir.glob("*.exe")) + list(benign_dir.glob("*.dll"))

    console.print(f"[green]✅ Found {len(malware_files)} malware samples[/green]")
    console.print(f"[green]✅ Found {len(benign_files)} benign samples[/green]")

    if len(malware_files) == 0 or len(benign_files) == 0:
        raise ValueError("Need both malware and benign samples for training")

    # Extract features
    console.print("[yellow]📊 Extracting features...[/yellow]")

    features_list = []
    labels_list = []

    # Process malware samples
    for file_path in malware_files:
        try:
            features = feature_extractor.extract_concatenated_features(file_path)
            features_list.append(features)
            labels_list.append(1)  # Malware = 1
        except Exception as e:
            console.print(f"[yellow]⚠️  Skipping {file_path}: {e}[/yellow]")

    # Process benign samples
    for file_path in benign_files:
        try:
            features = feature_extractor.extract_concatenated_features(file_path)
            features_list.append(features)
            labels_list.append(0)  # Benign = 0
        except Exception as e:
            console.print(f"[yellow]⚠️  Skipping {file_path}: {e}[/yellow]")

    if len(features_list) == 0:
        raise ValueError("No valid samples could be processed")

    # Convert to numpy arrays
    features_array = np.array(features_list, dtype=np.float32)
    labels_array = np.array(labels_list, dtype=np.int64)

    console.print(f"[green]✅ Processed {len(features_array)} samples[/green]")
    console.print(f"[green]✅ Feature vector size: {features_array.shape[1]}[/green]")

    # Create dataset
    dataset = MalwareDataset(features_array, labels_array)

    # Split into train/validation
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_save_path: Path,
    reports_dir: Path,
    epochs: int = 40,
    learning_rate: float = 0.001,
    device: str = "cpu",
) -> None:
    """Train the malware detection model.

    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        model_save_path: Path to save the trained model
        reports_dir: Directory to save training reports
        epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        device: Device to train on
    """
    console.print(f"[bold blue]🚀 Starting training on {device}[/bold blue]")

    # Initialize model
    model = MultiHeadMalwareDetector()
    console.print(
        f"[green]✅ Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters[/green]"
    )

    # Train the model
    history = train_multi_head_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
    )

    # Save the model
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_save_path)
    console.print(f"[green]✅ Model saved to {model_save_path}[/green]")

    # Create training report
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Calculate final metrics
    final_train_loss = history["train_losses"][-1]
    final_val_loss = history["val_losses"][-1]

    metrics = {
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "best_val_loss": min(history["val_losses"]),
        "epochs_trained": len(history["train_losses"]),
    }

    create_training_report(history, metrics, reports_dir)
    console.print(f"[green]✅ Training report saved to {reports_dir}[/green]")


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train the multi-headed malware detection model")
    parser.add_argument(
        "data_dir", type=Path, help="Directory containing malware and benign samples"
    )
    parser.add_argument(
        "--model-save-path",
        type=Path,
        default=Path("models/malware_detector.pt"),
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports/"),
        help="Directory to save training reports",
    )
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate for optimization"
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"], help="Device to train on"
    )
    parser.add_argument(
        "--train-split", type=float, default=0.8, help="Fraction of data to use for training"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging level",
    )

    args = parser.parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    try:
        # Check if data directory exists
        if not args.data_dir.exists():
            console.print(f"[red]❌ Data directory {args.data_dir} does not exist[/red]")
            sys.exit(1)

        # Initialize feature extractor
        feature_extractor = MultiHeadFeatureExtractor()

        # Load dataset
        train_loader, val_loader = load_dataset(args.data_dir, feature_extractor, args.train_split)

        # Train model
        train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            model_save_path=args.model_save_path,
            reports_dir=args.reports_dir,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            device=args.device,
        )

        console.print("[bold green]🎉 Training completed successfully![/bold green]")

    except Exception as e:
        console.print(f"[red]❌ Training failed: {str(e)}[/red]")
        logger.error("Training failed", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
