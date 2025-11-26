"""
Utility functions for the Bangladesh Population Analysis Project.

This module provides helper functions for common operations
like file handling, formatting, and calculations.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration for the application.

    Args:
        log_file: Optional path to log file.

    Returns:
        Configured logger instance.
    """
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)
    return logger


def format_population(population: float) -> str:
    """
    Format population number for display.

    Args:
        population: Population number to format.

    Returns:
        Formatted string representation.
    """
    if population >= 1_000_000_000:
        return f"{population / 1_000_000_000:.2f} billion"
    elif population >= 1_000_000:
        return f"{population / 1_000_000:.2f} million"
    elif population >= 1_000:
        return f"{population / 1_000:.2f} thousand"
    return str(int(population))


def calculate_growth_rate(
    initial: float,
    final: float,
    years: int
) -> float:
    """
    Calculate compound annual growth rate (CAGR).

    Args:
        initial: Initial population value.
        final: Final population value.
        years: Number of years between measurements.

    Returns:
        Annual growth rate as percentage.
    """
    if initial <= 0 or years <= 0:
        return 0.0
    return ((final / initial) ** (1 / years) - 1) * 100


def calculate_doubling_time(growth_rate: float) -> float:
    """
    Calculate population doubling time using Rule of 70.

    Args:
        growth_rate: Annual growth rate as percentage.

    Returns:
        Doubling time in years.
    """
    if growth_rate <= 0:
        return float('inf')
    return 70 / growth_rate


def save_json(data: Any, filepath: Path) -> None:
    """
    Save data to JSON file.

    Args:
        data: Data to save.
        filepath: Path to output file.
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Data saved to {filepath}")


def load_json(filepath: Path) -> Any:
    """
    Load data from JSON file.

    Args:
        filepath: Path to JSON file.

    Returns:
        Loaded data.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_report_header() -> str:
    """
    Generate a header for reports.

    Returns:
        Formatted header string.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"""
{'=' * 70}
    BANGLADESH POPULATION ANALYSIS REPORT
    Generated: {timestamp}
{'=' * 70}
"""
    return header


def validate_year_range(start: int, end: int) -> bool:
    """
    Validate year range for analysis.

    Args:
        start: Start year.
        end: End year.

    Returns:
        True if valid, False otherwise.
    """
    current_year = datetime.now().year
    if start < 1900 or end > current_year + 100:
        logger.warning(f"Year range {start}-{end} may be invalid")
        return False
    if start >= end:
        logger.error("Start year must be less than end year")
        return False
    return True


def calculate_statistics(data: list) -> dict:
    """
    Calculate basic statistical measures for a dataset.

    Args:
        data: List of numerical values.

    Returns:
        Dictionary containing statistical measures.
    """
    if not data:
        return {}
    
    arr = np.array(data)
    return {
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "range": float(np.max(arr) - np.min(arr)),
    }