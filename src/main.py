#!/usr/bin/env python3
"""Main entry point for the Immune malware detection system."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Dict

from rich.console import Console
from rich.logging import RichHandler

import torch

from src.models.malware_detector import MultiHeadMalwareDetector
from src.data.feature_extractor import MultiHeadFeatureExtractor

console = Console()


def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def analyze_file(file_path: Path, model_path: Optional[Path] = None) -> Dict[str, any]:
    """Analyze a file for malware detection.
    
    Args:
        file_path: Path to the file to analyze
        model_path: Optional path to a trained model
        
    Returns:
        Dictionary containing analysis results
    """
    console.print(f"[bold blue]🔍 Analyzing: {file_path}[/bold blue]")
    
    # Initialize feature extractor
    extractor = MultiHeadFeatureExtractor()
    
    try:
        # Extract features
        console.print("[yellow]📊 Extracting features...[/yellow]")
        features = extractor.extract_features(file_path)
        
        # Display feature information
        console.print(f"[green]✅ PE Features: {len(features['pe_features'])} dimensions[/green]")
        console.print(f"[green]✅ Byte Histogram: {len(features['byte_histogram'])} dimensions[/green]")
        console.print(f"[green]✅ Opcode Features: {len(features['opcode_features'])} dimensions[/green]")
        console.print(f"[green]✅ API Features: {len(features['api_features'])} dimensions[/green]")
        
        # If model is provided, make prediction
        if model_path and model_path.exists():
            console.print("[yellow]🤖 Loading model and making prediction...[/yellow]")
            
            # Load model
            model = MultiHeadMalwareDetector.load_model(model_path)
            model.eval()
            
            # Concatenate features for model input
            concatenated_features = extractor.extract_concatenated_features(file_path)
            features_tensor = torch.FloatTensor(concatenated_features).unsqueeze(0)
            
            # Make prediction
            with torch.no_grad():
                output = model(features_tensor)
                probabilities = torch.softmax(output, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][prediction].item()
            
            # Get feature importance
            feature_importance = model.get_feature_importance(features_tensor)
            
            result = {
                'file_path': str(file_path),
                'prediction': 'Malware' if prediction == 1 else 'Benign',
                'confidence': confidence,
                'malware_probability': probabilities[0][1].item(),
                'benign_probability': probabilities[0][0].item(),
                'feature_importance': {
                    'pe_latent': feature_importance['pe_latent'].numpy(),
                    'byte_latent': feature_importance['byte_latent'].numpy(),
                    'opcode_latent': feature_importance['opcode_latent'].numpy(),
                    'api_latent': feature_importance['api_latent'].numpy()
                }
            }
            
            # Display results
            prediction_color = "red" if prediction == 1 else "green"
            console.print(f"[bold {prediction_color}]🎯 Prediction: {result['prediction']}[/bold {prediction_color}]")
            console.print(f"[blue]📊 Confidence: {confidence:.2%}[/blue]")
            console.print(f"[blue]📊 Malware Probability: {result['malware_probability']:.2%}[/blue]")
            console.print(f"[blue]📊 Benign Probability: {result['benign_probability']:.2%}[/blue]")
            
        else:
            # No model provided, just show feature extraction
            result = {
                'file_path': str(file_path),
                'features': features,
                'message': 'Features extracted successfully. No model provided for prediction.'
            }
            console.print("[green]✅ Feature extraction completed successfully[/green]")
            console.print("[yellow]⚠️  No trained model provided - cannot make prediction[/yellow]")
        
        return result
        
    except Exception as e:
        error_msg = f"Error analyzing file {file_path}: {str(e)}"
        console.print(f"[red]❌ {error_msg}[/red]")
        logging.error(error_msg, exc_info=True)
        return {'error': error_msg}


def main() -> None:
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="Immune: ML-based malware detection system"
    )
    parser.add_argument(
        "--dev", action="store_true", help="Run in development mode"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging level",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Path to trained model file (.pt)",
    )
    parser.add_argument(
        "file_path",
        nargs="?",
        type=Path,
        help="Path to file to analyze",
    )

    args = parser.parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    if args.dev:
        logger.info("Running in development mode")
        console.print("[bold green]🚀 Immune Development Server Started[/bold green]")
        console.print("[yellow]Ready for malware analysis...[/yellow]")
        
        # Show model architecture info
        console.print("\n[bold blue]🏗️  Multi-Headed Model Architecture:[/bold blue]")
        console.print("• Head 1: PE Header Features (50 → 64 → 32)")
        console.print("• Head 2: Byte Histogram (256 → 256 → 128)")
        console.print("• Head 3: Opcode Features (300 → 256 → 128)")
        console.print("• Head 4: API Call Features (200 → 128 → 64)")
        console.print("• Fusion: 352 → 256 → 128 → 2 (output)")
        
        return

    if not args.file_path:
        console.print("[red]Error: No file path provided[/red]")
        parser.print_help()
        sys.exit(1)

    if not args.file_path.exists():
        console.print(f"[red]Error: File {args.file_path} does not exist[/red]")
        sys.exit(1)

    # Analyze the file
    result = analyze_file(args.file_path, args.model)
    
    if 'error' in result:
        sys.exit(1)
    
    console.print("[green]✅ File analysis completed[/green]")


if __name__ == "__main__":
    main()
