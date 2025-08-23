"""Neural network models for malware detection."""

from .malware_detector import MalwareDataset, MalwareDetector, evaluate_model, train_model

__all__ = ["MalwareDetector", "MalwareDataset", "train_model", "evaluate_model"]
