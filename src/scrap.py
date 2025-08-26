"""
Scrap malware samples from MalwareBazaar
"""

import argparse
import sys
from pathlib import Path

# Add the Immune package to the path
sys.path.append(str(Path(__file__).parent / "Immune"))

from Immune.data.scrapper import MalwareScraper


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Download malware samples from MalwareBazaar",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scrap.py                           # Download 1000 samples to data/malware/
  python scrap.py -o my_samples            # Download to my_samples/ directory
  python scrap.py -l 500                   # Download only 500 samples
  python scrap.py -o data/malware -l 200   # Download 200 samples to data/malware/ directory
        """,
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="data/malware",
        help="Output directory for downloaded samples (default: data/malware)",
    )

    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=1000,
        help="Maximum number of samples to download (default: 1000)",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", default=False, help="Enable verbose output"
    )

    return parser.parse_args()


def main():
    """Main function to download malware samples"""

    # Parse command line arguments
    args = parse_arguments()

    # Create output directory
    output_dir = Path(args.output)

    # Download malware samples
    print(f"Downloading malware samples to {output_dir}")
    print(f"Sample limit: {args.limit}")

    scraper = MalwareScraper(output_dir=output_dir, limit=args.limit)
    scraper.download_malware_samples()

    print("Download completed!")


if __name__ == "__main__":
    main()
