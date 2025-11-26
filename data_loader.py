"""
Data loading module for Bangladesh Population Analysis Project.

This module handles data acquisition from various sources including
APIs and local CSV files.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

from config import DATA_DIR, DATA_SOURCES
from utils import logger, save_json


class DataLoader:
    """
    Class responsible for loading and acquiring population data.
    
    This class handles data retrieval from World Bank API and
    local data sources for Bangladesh population analysis.
    """

    def __init__(self, cache_enabled: bool = True):
        """
        Initialize the DataLoader.

        Args:
            cache_enabled: Whether to cache downloaded data locally.
        """
        self.cache_enabled = cache_enabled
        self.cache_dir = DATA_DIR
        self._session = requests.Session()
        logger.info("DataLoader initialized")

    def fetch_world_bank_data(
        self,
        indicator_url: str,
        indicator_name: str
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data from World Bank API.

        Args:
            indicator_url: World Bank API URL for the indicator.
            indicator_name: Name of the indicator for logging.

        Returns:
            DataFrame with year and value columns, or None if failed.
        """
        cache_file = self.cache_dir / f"{indicator_name}_cache.json"
        
        # Check cache first
        if self.cache_enabled and cache_file.exists():
            logger.info(f"Loading {indicator_name} from cache")
            try:
                df = pd.read_json(cache_file)
                return df
            except Exception as e:
                logger.warning(f"Cache read failed: {e}")

        logger.info(f"Fetching {indicator_name} from World Bank API")
        
        try:
            response = self._session.get(indicator_url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if len(data) < 2 or data[1] is None:
                logger.error(f"No data returned for {indicator_name}")
                return None

            # Parse the World Bank API response
            records = []
            for item in data[1]:
                if item['value'] is not None:
                    records.append({
                        'year': int(item['date']),
                        'value': float(item['value'])
                    })
            
            df = pd.DataFrame(records)
            df = df.sort_values('year').reset_index(drop=True)
            
            # Cache the data
            if self.cache_enabled:
                df.to_json(cache_file, orient='records', indent=2)
                logger.info(f"Cached {indicator_name} data")
            
            return df

        except requests.RequestException as e:
            logger.error(f"API request failed for {indicator_name}: {e}")
            return None
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Data parsing failed for {indicator_name}: {e}")
            return None

    def load_all_indicators(self) -> Dict[str, pd.DataFrame]:
        """
        Load all population-related indicators.

        Returns:
            Dictionary mapping indicator names to DataFrames.
        """
        indicators = {}
        
        indicator_mapping = {
            "world_bank_population": "population",
            "world_bank_density": "density",
            "world_bank_growth_rate": "growth_rate",
            "world_bank_urban_pop": "urban_population_pct",
            "world_bank_fertility": "fertility_rate",
            "world_bank_birth_rate": "birth_rate",
            "world_bank_death_rate": "death_rate",
        }
        
        for source_key, indicator_name in indicator_mapping.items():
            url = DATA_SOURCES.get(source_key)
            if url:
                df = self.fetch_world_bank_data(url, indicator_name)
                if df is not None:
                    indicators[indicator_name] = df
                # Rate limiting - be respectful to the API
                time.sleep(0.5)
        
        logger.info(f"Loaded {len(indicators)} indicators successfully")
        return indicators

    def create_combined_dataset(
        self,
        indicators: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Combine all indicators into a single DataFrame.

        Args:
            indicators: Dictionary of indicator DataFrames.

        Returns:
            Combined DataFrame with all indicators.
        """
        if not indicators:
            logger.error("No indicators provided for combination")
            return pd.DataFrame()

        # Start with the first indicator
        combined = None
        
        for name, df in indicators.items():
            df_renamed = df.rename(columns={'value': name})
            
            if combined is None:
                combined = df_renamed
            else:
                combined = pd.merge(
                    combined,
                    df_renamed,
                    on='year',
                    how='outer'
                )
        
        combined = combined.sort_values('year').reset_index(drop=True)
        logger.info(f"Created combined dataset with {len(combined)} rows")
        
        return combined

    def load_division_population_data(self) -> pd.DataFrame:
        """
        Load or create division-wise population data for Bangladesh.
        
        Note: This uses estimated data based on census proportions.
        In a real scenario, you would load this from BBS website.

        Returns:
            DataFrame with division population estimates.
        """
        # 2022 Census estimates (approximate proportions)
        # Source: Bangladesh Bureau of Statistics
        division_data = {
            'Division': [
                'Dhaka', 'Chittagong', 'Rajshahi', 'Khulna',
                'Barisal', 'Sylhet', 'Rangpur', 'Mymensingh'
            ],
            'Population_2022': [
                44212152, 34566628, 20353041, 17415318,
                9323046, 11286036, 17610956, 12690190
            ],
            'Population_2011': [
                36054418, 28423019, 18484858, 15687759,
                8325666, 9910219, 15787758, 11370000
            ],
            'Area_km2': [
                20593, 33771, 18197, 22285,
                13297, 12596, 16317, 10584
            ],
        }
        
        df = pd.DataFrame(division_data)
        
        # Calculate derived metrics
        df['Density_2022'] = df['Population_2022'] / df['Area_km2']
        df['Growth_Rate'] = (
            (df['Population_2022'] / df['Population_2011']) ** (1/11) - 1
        ) * 100
        
        logger.info("Loaded division-wise population data")
        return df

    def export_data(
        self,
        df: pd.DataFrame,
        filename: str,
        format_type: str = 'csv'
    ) -> Path:
        """
        Export DataFrame to file.

        Args:
            df: DataFrame to export.
            filename: Output filename (without extension).
            format_type: Output format ('csv', 'json', or 'excel').

        Returns:
            Path to exported file.
        """
        output_path = DATA_DIR / f"{filename}.{format_type}"
        
        if format_type == 'csv':
            df.to_csv(output_path, index=False)
        elif format_type == 'json':
            df.to_json(output_path, orient='records', indent=2)
        elif format_type == 'excel':
            df.to_excel(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        logger.info(f"Exported data to {output_path}")
        return output_path