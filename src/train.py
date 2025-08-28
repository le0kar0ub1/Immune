import argparse
import json
import logging
import math
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
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


def load_features_from_json(features_file: Path) -> Tuple[Dict[str, Dict[str, Any]], int, int]:
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

        if "features" in data and "labels" in data:
            feature_nbr = len(data["features"][0])
            sample_nbr = len(data["features"])
        elif "feature_array" in data[0] and "is_malware" in data[0]:
            feature_nbr = len(data[0]["feature_array"])
            sample_nbr = len(data)
        else:
            raise ValueError("Invalid features format")

        console.print(f"[green]✅ Loaded {sample_nbr} samples[/green]")
        console.print(f"[green]✅ Feature vector size: {feature_nbr}[/green]")

        return data, feature_nbr, sample_nbr

    except Exception as e:
        raise ValueError(f"Failed to load features from {features_file}") from e


def prepare_data_loader(
    features: Dict[str, Dict[str, Any]],
    batch_size: int = 256,
    shuffle: bool = True,
    cut_off: int = None,
) -> DataLoader:
    """Prepare data loader for training, validation, and testing.

    Args:
        features: Feature array
        labels: Label array
        batch_size: Batch size for data loaders
        shuffle: Whether to shuffle data

    Returns:
        DataLoader
    """
    dataset = MalwareDataset(features, cut_off=cut_off)
    console.print(f"[green]✅ Dataset size: {len(dataset)}[/green]")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def prepare_data_loaders(
    features: Dict[str, Dict[str, Any]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 256,
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
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    console.print(
        f"[green]✅ Data split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}[/green]"
    )

    return train_loader, val_loader, test_loader


def generate_training_setups(max_runs=40, seed=42):
    """
    Yield up to max_runs setups of (time_minutes, hidden_size, n_hidden_layers).
    Uses a coarse grid, then samples evenly to cover the space.
    """
    random.seed(seed)

    # candidate grid
    # time_choices = [1, 1]
    time_choices = [2, 5, 10]
    hidden_size_choices = [64, 128, 256, 512]
    n_hidden_layers = [1, 3, 5, 7, 10]

    candidates = list(product(time_choices, hidden_size_choices, n_hidden_layers))
    candidates.sort()

    total = len(candidates)
    if total <= max_runs:
        chosen = candidates
    else:
        # evenly spaced picks
        indices = [math.floor(i * (total / max_runs)) for i in range(max_runs)]
        chosen = [candidates[i] for i in indices]

    for time_m, hidden, n_layers in chosen:
        yield {
            "time_limit_in_seconds": time_m * 60,
            "hidden_size": hidden,
            "n_hidden_layers": n_layers,
            "setup_signature": f"time_m{time_m}_hidden{hidden}_n_layers{n_layers}",
        }


def run_dnn_training_pipeline_with_setups(
    features_file: Path,
    features_file_valtest: Path,
    model_save_path: Path,
    reports_dir: Path,
    epochs: int = 100,
    learning_rate: float = 0.001,
    device: str = "cuda",
    batch_size: int = 1024,
    weight_decay: float = 1e-5,
    max_parallel_workers: int = 5,
) -> Dict[str, List[float]]:
    """Run the complete training pipeline using pre-computed features."""
    console.print("[bold blue]🚀 Starting complete training pipeline...[/bold blue]")

    # Load pre-computed features
    features, feature_nbr, sample_nbr = load_features_from_json(features_file)
    features_valtest, feature_nbr_valtest, sample_nbr_valtest = load_features_from_json(
        features_file_valtest
    )
    # features, feature_nbr, sample_nbr = load_features_from_json(Path("data/ember/train_features_sample.json"))
    # features_valtest, feature_nbr_valtest, sample_nbr_valtest = load_features_from_json(Path("data/ember/train_features_sample.json"))

    # Prepare data loaders

    # Prepare data loaders
    train_loader = prepare_data_loader(features, batch_size=batch_size, shuffle=True)
    val_loader = prepare_data_loader(features_valtest, batch_size=batch_size, shuffle=False)

    # Initialize scaling laws collector
    from Immune.utils.scaling_laws_collector import get_scaling_collector

    scaling_collector = get_scaling_collector(reports_dir / "scaling_laws")

    def train_single_setup(setup):
        """Train a single model setup."""
        try:
            console.print(
                f"[bold blue]🚀 Starting training on {device} with setup: {setup}[/bold blue]"
            )

            # Start scaling laws collection for this run
            run_name = setup["setup_signature"]
            scaling_collector.start_run(
                run_name=run_name,
                model_type="DNN",
                layers=[feature_nbr, setup["hidden_size"] * setup["n_hidden_layers"], 1],
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                batch_size=batch_size,
                epochs=epochs,
                time_limit_seconds=setup["time_limit_in_seconds"],
                feature_count=feature_nbr,
                train_samples=sample_nbr,
                val_samples=sample_nbr_valtest,
                test_samples=sample_nbr_valtest,
                device=device,
                total_parameters=0,  # Will be updated after model creation
                trainable_parameters=0,  # Will be updated after model creation
            )

            model = DNNMalwareDetector(
                layers=[feature_nbr, *([setup["hidden_size"]] * setup["n_hidden_layers"]), 1]
            )

            # Update scaling collector with model parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            scaling_collector.update_run_metrics(
                total_parameters=total_params, trainable_parameters=trainable_params
            )

            console.print(f"[green]✅ Model initialized with {total_params:,} parameters[/green]")

            # Track training start time
            training_start_time = time.time()

            # Train the model (disable graph logging to avoid TensorBoard conflicts during parallel execution)
            history = train_dnn_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                learning_rate=learning_rate,
                device=device,
                weight_decay=weight_decay,
                time_limit_in_seconds=setup["time_limit_in_seconds"],
                run_name=setup["setup_signature"],
                log_graph=True,
                verbose_training=False,
            )

            # Calculate actual training time
            actual_training_time = time.time() - training_start_time

            # Update scaling collector with training results
            # Add safety checks for history data
            if history and "train_losses" in history and history["train_losses"]:
                final_train_loss = history["train_losses"][-1]
                epochs_trained = len(history["train_losses"])
            else:
                final_train_loss = None
                epochs_trained = None

            if history and "val_losses" in history and history["val_losses"]:
                final_val_loss = history["val_losses"][-1]
                best_val_loss = min(history["val_losses"])
            else:
                final_val_loss = None
                best_val_loss = None

            scaling_collector.update_run_metrics(
                final_train_loss=final_train_loss,
                final_val_loss=final_val_loss,
                best_val_loss=best_val_loss,
                epochs_trained=epochs_trained,
                actual_training_time=actual_training_time,
            )

            # Complete the run in scaling collector
            scaling_collector.complete_run(training_history=history)

            console.print(
                f"[green]✅ Training completed in {setup['time_limit_in_seconds']} seconds[/green]"
            )

            # Create training report with metrics extracted from history
            if history and "train_losses" in history and "val_losses" in history:
                report_metrics = {
                    "final_train_loss": history["train_losses"][-1]
                    if history["train_losses"]
                    else 0.0,
                    "final_val_loss": history["val_losses"][-1] if history["val_losses"] else 0.0,
                    "best_val_loss": min(history["val_losses"]) if history["val_losses"] else 0.0,
                    "epochs_trained": len(history["train_losses"])
                    if history["train_losses"]
                    else 0,
                    "training_time_seconds": actual_training_time,
                    "model_parameters": total_params,
                    "hidden_size": setup["hidden_size"],
                    "n_hidden_layers": setup["n_hidden_layers"],
                    "time_limit_seconds": setup["time_limit_in_seconds"],
                }

                # Create reports directory for this setup
                setup_reports_dir = reports_dir / f"{setup['setup_signature']}_reports"
                create_training_report(history, report_metrics, setup_reports_dir)
            else:
                console.print(
                    "[yellow]⚠️  No training history available for report creation[/yellow]"
                )

            return setup["setup_signature"], True, None
        except Exception as e:
            console.print(
                f"[red]❌ Training failed for setup {setup['setup_signature']}: {str(e)}[/red]"
            )
            return setup["setup_signature"], False, str(e)

    # Get all setups and run them in parallel
    setups = list(generate_training_setups())
    console.print(
        f"[bold blue]🚀 Starting parallel training for {len(setups)} setups...[/bold blue]"
    )

    # Use ThreadPoolExecutor for parallel execution
    # Note: Using threads instead of processes to avoid GPU memory issues
    # GPU memory is shared between threads, so limit concurrent runs
    if max_parallel_workers is None:
        max_workers = min(len(setups), 2 if device == "cuda" else 4)  # More conservative for GPU
    else:
        max_workers = min(len(setups), max_parallel_workers)

    console.print(
        f"[yellow]⚠️  Using {max_workers} parallel workers (limited for {'GPU' if device == 'cuda' else 'CPU'} memory)[/yellow]"
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all training tasks
        future_to_setup = {executor.submit(train_single_setup, setup): setup for setup in setups}

        # Process completed tasks
        completed_count = 0
        failed_count = 0

        for future in as_completed(future_to_setup):
            setup_name, success, error = future.result()
            completed_count += 1
            if success:
                console.print(
                    f"[green]✅ Setup {setup_name} completed successfully ({completed_count}/{len(setups)})[/green]"
                )
            else:
                failed_count += 1
                console.print(
                    f"[red]❌ Setup {setup_name} failed: {error} ({completed_count}/{len(setups)})[/red]"
                )

            # Progress update
            remaining = len(setups) - completed_count
            if remaining > 0:
                console.print(
                    f"[blue]📊 Progress: {completed_count}/{len(setups)} completed, {remaining} remaining[/blue]"
                )

    # Final summary
    console.print("[bold green]🎉 Parallel training completed![/bold green]")
    console.print(f"[green]✅ Successful: {completed_count - failed_count}[/green]")
    if failed_count > 0:
        console.print(f"[red]❌ Failed: {failed_count}[/red]")

    # Export scaling laws data for plotting
    scaling_collector.export_for_plotting()
    console.print("[bold green]📊 Scaling laws data exported for analysis![/bold green]")


def run_dnn_training_pipeline(
    features_file: Path,
    features_file_valtest: Path,
    model_save_path: Path,
    reports_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    epochs: int = 100,
    learning_rate: float = 0.001,
    device: str = "cuda",
    batch_size: int = 256,
    early_stopping_patience: int = 15,
    weight_decay: float = 1e-5,
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
    features, feature_nbr, sample_nbr = load_features_from_json(features_file)
    features_valtest, feature_nbr_valtest, sample_nbr_valtest = load_features_from_json(
        features_file_valtest
    )

    # Prepare data loaders
    train_loader = prepare_data_loader(features, batch_size=batch_size, shuffle=True)
    # cut_off = sample_nbr_valtest // 2
    val_loader = prepare_data_loader(features_valtest, batch_size=batch_size, shuffle=False)
    test_loader = prepare_data_loader(features_valtest, batch_size=batch_size, shuffle=False)

    # # Prepare data loaders
    # train_loader, val_loader, test_loader = prepare_data_loaders(
    #     features, train_ratio, val_ratio, batch_size, shuffle_train=True
    # )

    # Initialize model using existing DNNMalwareDetector class
    # layers=[feature_nbr, feature_nbr // 2, feature_nbr // 4, feature_nbr // 8, feature_nbr // 16, 1]
    # layers=[feature_nbr, feature_nbr // 4, feature_nbr // 16, 1]
    model = DNNMalwareDetector()
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
        weight_decay=weight_decay,
        time_limit_in_seconds=600,
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


def run_dnn_evaluation_pipeline(
    features_file_valtest: Path,
    model_save_path: Path,
    reports_dir: Path,
    device: str = "cuda",
) -> Dict[str, List[float]]:
    """Run the complete evaluation pipeline using pre-computed features."""
    console.print("[bold blue]🚀 Starting complete evaluation pipeline...[/bold blue]")

    # Load pre-computed features
    features_valtest, feature_nbr_valtest, sample_nbr_valtest = load_features_from_json(
        features_file_valtest
    )

    # Prepare data loaders
    test_loader = prepare_data_loader(features_valtest, batch_size=256, shuffle=True)

    # Load the model
    model = DNNMalwareDetector.load_model(model_save_path, device=device)

    # Evaluate model using existing evaluate_dnn_model function
    console.print("[yellow]📊 Evaluating model on test set...[/yellow]")
    metrics = evaluate_dnn_model(model, test_loader, device=device)

    # Create training report
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Calculate final metrics for report
    {
        "test_loss": metrics["test_loss"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "roc_auc": metrics["roc_auc"],
    }

    # create_training_report(None, report_metrics, reports_dir)

    console.print("[bold green]🎉 Evaluation pipeline completed successfully![/bold green]")
    return metrics


def run_xgb_training_pipeline(
    features_file: Path,
    features_file_valtest: Path,
    model_save_path: Path,
    reports_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Dict[str, List[float]]:
    """Run the complete training pipeline using pre-computed features."""
    console.print("[bold blue]🚀 Starting complete training pipeline...[/bold blue]")

    # Load pre-computed features
    features, feature_nbr, sample_nbr = load_features_from_json(features_file)
    features_valtest, feature_nbr_valtest, sample_nbr_valtest = load_features_from_json(
        features_file_valtest
    )

    # Prepare data loaders
    # train_loader, val_loader, test_loader = prepare_data_loaders(features, train_ratio, val_ratio)
    train_loader = prepare_data_loader(features, batch_size=64, shuffle=True)
    val_loader = prepare_data_loader(
        features_valtest, batch_size=64, shuffle=False, cut_off=-(sample_nbr_valtest // 2)
    )
    test_loader = prepare_data_loader(
        features_valtest, batch_size=64, shuffle=False, cut_off=sample_nbr_valtest // 2
    )

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
        default=Path("data/ember/train_features_1.json"),
        help="Path to features.json file containing pre-computed features",
    )
    parser.add_argument(
        "--features-file-valtest",
        type=Path,
        default=Path("data/ember/test_features.json"),
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
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate for optimization"
    )
    parser.add_argument(
        "--device", default="cuda", choices=["cpu", "cuda"], help="Device to train on"
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8, help="Fraction of data to use for training"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1, help="Fraction of data to use for validation"
    )
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for training")
    parser.add_argument(
        "--early-stopping-patience", type=int, default=15, help="Patience for early stopping"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging level",
    )
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument(
        "--max-parallel-workers",
        type=int,
        default=5,
        help="Maximum number of parallel workers for training (default: auto-detect based on device)",
    )
    parser.add_argument(
        "--parallel-setups",
        action="store_true",
        help="Run multiple training setups in parallel (scaling laws mode)",
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

        device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"

        # Run training pipeline
        console.print("[bold blue]🚀 Starting training with hyperparameters:[/bold blue]")
        console.print(f"[bold blue]     Features file: {args.features_file}[/bold blue]")
        console.print(f"[bold blue]     Model save path: {args.model_save_path}[/bold blue]")
        console.print(f"[bold blue]     Reports dir: {args.reports_dir}[/bold blue]")
        console.print(f"[bold blue]     Train ratio: {args.train_ratio}[/bold blue]")
        console.print(f"[bold blue]     Val ratio: {args.val_ratio}[/bold blue]")
        console.print(f"[bold blue]     Epochs: {args.epochs}[/bold blue]")
        console.print(f"[bold blue]     Learning rate: {args.learning_rate}[/bold blue]")
        console.print(f"[bold blue]     Device: {device}[/bold blue]")
        console.print(f"[bold blue]     Batch size: {args.batch_size}[/bold blue]")
        console.print(
            f"[bold blue]     Early stopping patience: {args.early_stopping_patience}[/bold blue]"
        )

        # if args.evaluate:
        #     run_dnn_evaluation_pipeline(
        #         features_file_valtest=args.features_file_valtest,
        #         model_save_path=args.model_save_path,
        #         reports_dir=args.reports_dir,
        #         device=device,
        #     )
        # elif args.parallel_setups:
        #     console.print(
        #         f"[bold blue]🚀 Running parallel setups mode with max workers: {args.max_parallel_workers or 'auto'}[/bold blue]"
        #     )
        #     run_dnn_training_pipeline_with_setups(
        #         features_file=args.features_file,
        #         features_file_valtest=args.features_file_valtest,
        #         model_save_path=args.model_save_path,
        #         reports_dir=args.reports_dir,
        #         epochs=args.epochs,
        #         learning_rate=args.learning_rate,
        #         device=device,
        #         batch_size=args.batch_size,
        #         weight_decay=1e-5,
        #         max_parallel_workers=args.max_parallel_workers,
        #     )
        # else:
        #     run_dnn_training_pipeline(
        #         features_file=args.features_file,
        #         features_file_valtest=args.features_file_valtest,
        #         model_save_path=args.model_save_path,
        #         reports_dir=args.reports_dir,
        #         train_ratio=args.train_ratio,
        #         val_ratio=args.val_ratio,
        #         epochs=args.epochs,
        #         learning_rate=args.learning_rate,
        #         device=device,
        #         batch_size=args.batch_size,
        #         early_stopping_patience=args.early_stopping_patience,
        #     )

        run_xgb_training_pipeline(
            features_file=args.features_file,
            features_file_valtest=args.features_file_valtest,
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
