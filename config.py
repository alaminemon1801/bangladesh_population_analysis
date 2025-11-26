"""
Configuration settings for Bangladesh Population Analysis Project.

This module contains all configuration constants and settings
used throughout the application.
"""

import os
from pathlib import Path

# Project directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Data source URLs (Bangladesh Bureau of Statistics and World Bank)
DATA_SOURCES = {
    "world_bank_population": (
        "https://api.worldbank.org/v2/country/BGD/indicator/"
        "SP.POP.TOTL?format=json&per_page=100"
    ),
    "world_bank_density": (
        "https://api.worldbank.org/v2/country/BGD/indicator/"
        "EN.POP.DNST?format=json&per_page=100"
    ),
    "world_bank_growth_rate": (
        "https://api.worldbank.org/v2/country/BGD/indicator/"
        "SP.POP.GROW?format=json&per_page=100"
    ),
    "world_bank_urban_pop": (
        "https://api.worldbank.org/v2/country/BGD/indicator/"
        "SP.URB.TOTL.IN.ZS?format=json&per_page=100"
    ),
    "world_bank_fertility": (
        "https://api.worldbank.org/v2/country/BGD/indicator/"
        "SP.DYN.TFRT.IN?format=json&per_page=100"
    ),
    "world_bank_birth_rate": (
        "https://api.worldbank.org/v2/country/BGD/indicator/"
        "SP.DYN.CBRT.IN?format=json&per_page=100"
    ),
    "world_bank_death_rate": (
        "https://api.worldbank.org/v2/country/BGD/indicator/"
        "SP.DYN.CDRT.IN?format=json&per_page=100"
    ),
}

# Bangladesh geographic data
BANGLADESH_AREA_KM2 = 147570  # Square kilometers

# Division-wise area data (approximate)
DIVISION_AREAS = {
    "Dhaka": 20593,
    "Chittagong": 33771,
    "Rajshahi": 18197,
    "Khulna": 22285,
    "Barisal": 13297,
    "Sylhet": 12596,
    "Rangpur": 16317,
    "Mymensingh": 10584,
}

# Analysis parameters
PROJECTION_YEARS = 30  # Years to project into the future
ANALYSIS_START_YEAR = 1960
ANALYSIS_END_YEAR = 2023

# Visualization settings
FIGURE_SIZE = (12, 8)
PLOT_STYLE = "seaborn-v0_8-whitegrid"
COLOR_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"
]