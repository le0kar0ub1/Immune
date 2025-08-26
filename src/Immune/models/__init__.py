"""Neural network models for malware detection."""

from .dnn_malware_detector import MalwareDataset, DNNMalwareDetector, evaluate_dnn_model, train_dnn_model
from .xgb_malware_detector import train_xgb_model

__all__ = ["DNNMalwareDetector", "MalwareDataset", "train_dnn_model", "evaluate_dnn_model", "train_xgb_model"]
