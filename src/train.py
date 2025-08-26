import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from rich.console import Console
from rich.logging import RichHandler
from torch.utils.data import DataLoader, random_split

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from Immune.models.dnn_malware_detector import (
    DNNMalwareDetector,
    MalwareDataset,
    evaluate_dnn_model,
    train_dnn_model,
)
from Immune.models.xgb_malware_detector import train_xgb_model
from Immune.utils.visualization import create_training_report, create_xgb_training_report

console = Console()


def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def load_features_from_json(features_file: Path) -> Tuple[Dict[str, Dict[str, Any]], int]:
    """Load pre-computed features from JSON file.

    Args:
        features_file: Path to features.json file

    Returns:
        Tuple of (features, labels) arrays
    """
    console.print(f"[yellow]📁 Loading pre-computed features from {features_file}...[/yellow]")

    if not features_file.exists():
        raise FileNotFoundError(f"Features file {features_file} not found")

    try:
        with open(features_file, "r") as f:
            data = json.load(f)

        feature_nbr = len(data[list(data.keys())[0]]["feature_array"])
        console.print(f"[green]✅ Loaded {len(data.keys())} samples[/green]")
        console.print(f"[green]✅ Feature vector size: {feature_nbr}[/green]")

        return data, feature_nbr

    except Exception as e:
        raise ValueError(f"Failed to load features from {features_file}") from e


def prepare_data_loaders(
    features: Dict[str, Dict[str, Any]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 64,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Prepare data loaders for training, validation, and testing.

    Args:
        features: Feature array
        labels: Label array
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        batch_size: Batch size for data loaders
        shuffle_train: Whether to shuffle training data

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    dataset = MalwareDataset(features)

    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    console.print(
        f"[green]✅ Data split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}[/green]"
    )

    return train_loader, val_loader, test_loader


def run_dnn_training_pipeline(
    features_file: Path,
    model_save_path: Path,
    reports_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    epochs: int = 100,
    learning_rate: float = 0.001,
    device: str = "cpu",
    batch_size: int = 64,
    early_stopping_patience: int = 15,
) -> Dict[str, List[float]]:
    """Run the complete training pipeline using pre-computed features.

    Args:
        features_file: Path to features.json file
        model_save_path: Path to save the trained model
        reports_dir: Directory to save training reports
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        device: Device to train on
        batch_size: Batch size for training
        early_stopping_patience: Patience for early stopping

    Returns:
        Training history dictionary
    """
    console.print("[bold blue]🚀 Starting complete training pipeline...[/bold blue]")

    # Load pre-computed features
    features, feature_nbr = load_features_from_json(features_file)

    # Prepare data loaders
    train_loader, val_loader, test_loader = prepare_data_loaders(
        features, train_ratio, val_ratio, batch_size
    )

    # Initialize model using existing DNNMalwareDetector class
    model = DNNMalwareDetector(layers=[feature_nbr, 128, 64, 1])
    console.print(
        f"[green]✅ Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters[/green]"
    )

    # Train model using existing train_dnn_model function
    console.print(f"[bold blue]🚀 Starting training on {device}[/bold blue]")
    history = train_dnn_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        early_stopping_patience=early_stopping_patience,
    )

    # Save the model using existing save_model method
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_save_path)
    console.print(f"[green]✅ Model saved to {model_save_path}[/green]")

    # Evaluate model using existing evaluate_dnn_model function
    console.print("[yellow]📊 Evaluating model on test set...[/yellow]")
    metrics = evaluate_dnn_model(model, test_loader, device)

    # Create training report
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Calculate final metrics for report
    final_train_loss = history["train_losses"][-1]
    final_val_loss = history["val_losses"][-1]

    report_metrics = {
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "best_val_loss": min(history["val_losses"]),
        "epochs_trained": len(history["train_losses"]),
        **metrics,  # Include evaluation metrics
    }

    create_training_report(history, report_metrics, reports_dir)

    console.print("[bold green]🎉 Training pipeline completed successfully![/bold green]")
    return history


def run_xgb_training_pipeline(
    features_file: Path,
    model_save_path: Path,
    reports_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Dict[str, List[float]]:
    """Run the complete training pipeline using pre-computed features."""
    console.print("[bold blue]🚀 Starting complete training pipeline...[/bold blue]")

    # Load pre-computed features
    features, feature_nbr = load_features_from_json(features_file)

    # Prepare data loaders
    train_loader, val_loader, test_loader = prepare_data_loaders(features, train_ratio, val_ratio)

    # Train model using existing train_xgb_model function
    console.print("[bold blue]🚀 Starting training XGBoost model[/bold blue]")
    results = train_xgb_model(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    create_xgb_training_report(results, reports_dir)

    console.print("[bold green]🎉 Training pipeline completed successfully![/bold green]")
    return results


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train the malware detection model using pre-computed features"
    )
    parser.add_argument(
        "--features-file",
        type=Path,
        default=Path("data/formatted_features.json"),
        help="Path to features.json file containing pre-computed features",
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
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument(
        "--learning-rate", type=float, default=0.01, help="Learning rate for optimization"
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"], help="Device to train on"
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8, help="Fraction of data to use for training"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1, help="Fraction of data to use for validation"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument(
        "--early-stopping-patience", type=int, default=15, help="Patience for early stopping"
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
        # Check if features file exists
        if not args.features_file.exists():
            console.print(f"[red]❌ Features file {args.features_file} does not exist[/red]")
            console.print(
                "[yellow]💡 Make sure to run feature extraction first to generate data/features.json[/yellow]"
            )
            sys.exit(1)

        # Run training pipeline
        console.print("[bold blue]🚀 Starting training with hyperparameters:[/bold blue]")
        console.print(f"[bold blue]     Features file: {args.features_file}[/bold blue]")
        console.print(f"[bold blue]     Model save path: {args.model_save_path}[/bold blue]")
        console.print(f"[bold blue]     Reports dir: {args.reports_dir}[/bold blue]")
        console.print(f"[bold blue]     Train ratio: {args.train_ratio}[/bold blue]")
        console.print(f"[bold blue]     Val ratio: {args.val_ratio}[/bold blue]")
        console.print(f"[bold blue]     Epochs: {args.epochs}[/bold blue]")
        console.print(f"[bold blue]     Learning rate: {args.learning_rate}[/bold blue]")
        console.print(f"[bold blue]     Device: {args.device}[/bold blue]")
        console.print(f"[bold blue]     Batch size: {args.batch_size}[/bold blue]")
        console.print(
            f"[bold blue]     Early stopping patience: {args.early_stopping_patience}[/bold blue]"
        )

        # run_dnn_training_pipeline(
        #     features_file=args.features_file,
        #     model_save_path=args.model_save_path,
        #     reports_dir=args.reports_dir,
        #     train_ratio=args.train_ratio,
        #     val_ratio=args.val_ratio,
        #     epochs=args.epochs,
        #     learning_rate=args.learning_rate,
        #     device=args.device,
        #     batch_size=args.batch_size,
        #     early_stopping_patience=args.early_stopping_patience,
        # )

        run_xgb_training_pipeline(
            features_file=args.features_file,
            model_save_path=args.model_save_path,
            reports_dir=args.reports_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )

    except Exception as e:
        console.print(f"[red]❌ Training failed: {str(e)}[/red]")
        logger.error("Training failed", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
