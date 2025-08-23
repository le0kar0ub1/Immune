"""Main entry point for the malware detection system."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from utils.visualization import (
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_training_history,
)

from data.feature_extractor import FeatureExtractor
from models.malware_detector import MalwareDataset, MalwareDetector, evaluate_model, train_model

logger = logging.getLogger(__name__)


class MalwareDetectionPipeline:
    """Simple wrapper class for the complete malware detection workflow."""

    def __init__(self, model_config: Optional[Dict] = None):
        """Initialize the pipeline.

        Args:
            model_config: Optional configuration for the model
        """
        self.model_config = model_config or {
            "input_size": 356,
            "hidden_size": 128,
            "dropout_rate": 0.5,
        }

        self.feature_extractor = FeatureExtractor()
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Pipeline initialized on device: {self.device}")

    def extract_features_from_directory(
        self, data_dir: Path, labels: Dict[str, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features from all files in a directory.

        Args:
            data_dir: Directory containing binary files
            labels: Dictionary mapping filename to label (0=benign, 1=malware)

        Returns:
            Tuple of (features, labels) arrays
        """
        logger.info(f"Extracting features from {data_dir}")

        features_list = []
        labels_list = []

        for file_path in data_dir.glob("*"):
            if file_path.is_file():
                try:
                    features = self.feature_extractor.extract_features(file_path)
                    # Convert features to numpy array (you'll need to implement this)
                    feature_vector = self._features_to_vector(features)

                    features_list.append(feature_vector)
                    labels_list.append(labels.get(file_path.name, 0))

                except Exception as e:
                    logger.warning(f"Failed to extract features from {file_path}: {e}")
                    continue

        if not features_list:
            raise ValueError("No features could be extracted from the directory")

        return np.array(features_list), np.array(labels_list)

    def _features_to_vector(self, features) -> np.ndarray:
        """Convert features object to numpy vector. This is a placeholder."""
        return np.random.rand(356)

    def prepare_data(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare data loaders for training, validation, and testing.

        Args:
            features: Feature array
            labels: Label array
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        dataset = MalwareDataset(features, labels)

        # Calculate split sizes
        total_size = len(dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size

        # Split dataset
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        logger.info(
            f"Data split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}"
        )

        return train_loader, val_loader, test_loader

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        learning_rate: float = 0.001,
    ) -> Dict[str, List[float]]:
        """Train the malware detection model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization

        Returns:
            Training history dictionary
        """
        logger.info("Starting model training...")

        # Initialize model
        self.model = MalwareDetector(**self.model_config)

        # Train model
        history = train_model(
            model=self.model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            device=self.device,
        )

        logger.info("Training completed successfully")
        return history

    def test(self, test_loader: DataLoader) -> Dict[str, float]:
        """Test the trained model.

        Args:
            test_loader: Test data loader

        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before testing")

        logger.info("Starting model evaluation...")

        metrics = evaluate_model(model=self.model, test_loader=test_loader, device=self.device)

        logger.info("Evaluation completed successfully")
        return metrics

    def predict(self, features: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Make predictions on new data.

        Args:
            features: Feature array for prediction
            threshold: Classification threshold

        Returns:
            Array of predictions (0 or 1)
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")

        self.model.eval()
        features_tensor = torch.FloatTensor(features).to(self.device)

        with torch.no_grad():
            predictions = self.model.predict(features_tensor, threshold)

        return predictions.cpu().numpy()

    def visualize_results(
        self,
        history: Dict[str, List[float]],
        metrics: Dict[str, float],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray,
        save_dir: Path,
    ) -> None:
        """Create visualizations of the results.

        Args:
            history: Training history
            metrics: Evaluation metrics
            y_true: True labels
            y_pred: Predicted labels
            y_scores: Prediction scores
            save_dir: Directory to save visualizations
        """
        logger.info("Creating visualizations...")

        save_dir.mkdir(parents=True, exist_ok=True)

        # Training history
        plot_training_history(history, save_path=save_dir / "training_history.png", show_plot=False)

        # Confusion matrix
        plot_confusion_matrix(
            y_true, y_pred, save_path=save_dir / "confusion_matrix.png", show_plot=False
        )

        # ROC curve
        plot_roc_curve(y_true, y_scores, save_path=save_dir / "roc_curve.png", show_plot=False)

        # Precision-recall curve
        plot_precision_recall_curve(
            y_true, y_scores, save_path=save_dir / "precision_recall_curve.png", show_plot=False
        )

        # Save metrics
        metrics_file = save_dir / "metrics.txt"
        with open(metrics_file, "w") as f:
            f.write("Malware Detection Results\n")
            f.write("=" * 50 + "\n\n")
            for metric_name, value in metrics.items():
                f.write(f"{metric_name}: {value:.4f}\n")

        logger.info(f"Visualizations saved to {save_dir}")

    def save_model(self, filepath: Path) -> None:
        """Save the trained model.

        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")

        self.model.save_model(filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: Path) -> None:
        """Load a trained model.

        Args:
            filepath: Path to the saved model
        """
        self.model = MalwareDetector.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")

    def run_complete_pipeline(
        self, data_dir: Path, labels: Dict[str, int], output_dir: Path, epochs: int = 100
    ) -> Dict[str, float]:
        """Run the complete pipeline from start to finish.

        Args:
            data_dir: Directory containing binary files
            labels: Dictionary mapping filename to label
            output_dir: Directory to save results
            epochs: Number of training epochs

        Returns:
            Final evaluation metrics
        """
        logger.info("Starting complete malware detection pipeline...")

        # 1. Feature extraction
        features, labels_array = self.extract_features_from_directory(data_dir, labels)

        # 2. Data preparation
        train_loader, val_loader, test_loader = self.prepare_data(features, labels_array)

        # 3. Model training
        history = self.train(train_loader, val_loader, epochs=epochs)

        # 4. Model testing
        metrics = self.test(test_loader)

        # 5. Get predictions for visualization
        test_features = []
        test_labels = []
        test_scores = []

        self.model.eval()
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                batch_features = batch_features.to(self.device)
                outputs = self.model(batch_features)
                predictions = (outputs > 0.5).float()

                test_features.extend(batch_features.cpu().numpy())
                test_labels.extend(batch_labels.numpy())
                test_scores.extend(outputs.squeeze().cpu().numpy())

        # 6. Visualization
        self.visualize_results(
            history,
            metrics,
            np.array(test_labels),
            np.array(predictions),
            np.array(test_scores),
            output_dir,
        )

        # 7. Save model
        model_path = output_dir / "malware_detector.pth"
        self.save_model(model_path)

        logger.info("Pipeline completed successfully!")
        return metrics


def main():
    """Main function to demonstrate the pipeline usage."""
    # Example usage
    MalwareDetectionPipeline()

    # You would need to provide actual data directory and labels
    # data_dir = Path("path/to/your/binary/files")
    # labels = {"file1.exe": 0, "file2.exe": 1, ...}  # 0=benign, 1=malware
    # output_dir = Path("results")

    # metrics = pipeline.run_complete_pipeline(data_dir, labels, output_dir)
    # print("Final metrics:", metrics)


if __name__ == "__main__":
    main()
