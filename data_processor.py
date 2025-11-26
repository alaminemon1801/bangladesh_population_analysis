"""
Data processing module for Bangladesh Population Analysis Project.

This module contains classes and functions for processing,
analyzing, and transforming population data.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

from config import BANGLADESH_AREA_KM2, PROJECTION_YEARS
from utils import (
    calculate_doubling_time,
    calculate_growth_rate,
    calculate_statistics,
    format_population,
    logger,
)


@dataclass
class PopulationProjection:
    """Data class for storing population projections."""
    
    year: int
    projected_population: float
    lower_bound: float
    upper_bound: float
    model_type: str


class PopulationAnalyzer:
    """
    Class for analyzing population data and generating insights.
    
    This class provides methods for statistical analysis,
    trend detection, and population projections.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the PopulationAnalyzer.

        Args:
            data: DataFrame containing population data with 'year' column.
        """
        self.data = data.copy()
        self.analysis_results = {}
        logger.info("PopulationAnalyzer initialized")

    def calculate_basic_statistics(
        self,
        column: str = 'population'
    ) -> Dict[str, float]:
        """
        Calculate basic statistical measures for a data column.

        Args:
            column: Name of the column to analyze.

        Returns:
            Dictionary of statistical measures.
        """
        if column not in self.data.columns:
            logger.error(f"Column {column} not found in data")
            return {}

        values = self.data[column].dropna().tolist()
        stats_result = calculate_statistics(values)
        
        self.analysis_results[f'{column}_statistics'] = stats_result
        return stats_result

    def analyze_growth_trends(self) -> Dict[str, any]:
        """
        Analyze population growth trends over different periods.

        Returns:
            Dictionary containing growth trend analysis.
        """
        if 'population' not in self.data.columns:
            logger.error("Population column not found")
            return {}

        df = self.data.dropna(subset=['population']).copy()
        df = df.sort_values('year')

        # Calculate period-wise growth rates
        periods = [
            ("1960-1980", 1960, 1980),
            ("1980-2000", 1980, 2000),
            ("2000-2020", 2000, 2020),
            ("Overall", df['year'].min(), df['year'].max()),
        ]

        growth_analysis = {}
        
        for period_name, start, end in periods:
            start_data = df[df['year'] == start]
            end_data = df[df['year'] == end]
            
            if not start_data.empty and not end_data.empty:
                start_pop = start_data['population'].values[0]
                end_pop = end_data['population'].values[0]
                years = end - start
                
                growth_rate = calculate_growth_rate(start_pop, end_pop, years)
                doubling_time = calculate_doubling_time(growth_rate)
                
                growth_analysis[period_name] = {
                    'start_year': start,
                    'end_year': end,
                    'start_population': start_pop,
                    'end_population': end_pop,
                    'annual_growth_rate': round(growth_rate, 3),
                    'doubling_time_years': round(doubling_time, 1),
                    'absolute_increase': end_pop - start_pop,
                }

        self.analysis_results['growth_trends'] = growth_analysis
        return growth_analysis

    def calculate_density_analysis(self) -> Dict[str, float]:
        """
        Analyze population density trends.

        Returns:
            Dictionary with density analysis results.
        """
        if 'population' not in self.data.columns:
            return {}

        df = self.data.dropna(subset=['population']).copy()
        
        # Calculate density for each year
        df['calculated_density'] = df['population'] / BANGLADESH_AREA_KM2

        latest = df[df['year'] == df['year'].max()].iloc[0]
        earliest = df[df['year'] == df['year'].min()].iloc[0]

        density_analysis = {
            'current_density': round(latest['calculated_density'], 2),
            'earliest_density': round(earliest['calculated_density'], 2),
            'density_increase': round(
                latest['calculated_density'] - earliest['calculated_density'],
                2
            ),
            'density_increase_pct': round(
                (latest['calculated_density'] / earliest['calculated_density']
                 - 1) * 100,
                2
            ),
            'area_km2': BANGLADESH_AREA_KM2,
            'comparison_note': (
                "Bangladesh is one of the most densely populated countries "
                "in the world"
            ),
        }

        self.analysis_results['density_analysis'] = density_analysis
        return density_analysis

    @staticmethod
    def _logistic_growth(
        t: np.ndarray,
        carrying_capacity: float,
        growth_rate: float,
        midpoint: float
    ) -> np.ndarray:
        """
        Logistic growth function for population modeling.

        Args:
            t: Time values.
            carrying_capacity: Maximum population capacity.
            growth_rate: Growth rate parameter.
            midpoint: Midpoint of the logistic curve.

        Returns:
            Population values at given times.
        """
        return carrying_capacity / (1 + np.exp(-growth_rate * (t - midpoint)))

    def project_population(
        self,
        years_ahead: int = PROJECTION_YEARS,
        method: str = 'exponential'
    ) -> List[PopulationProjection]:
        """
        Project future population using different models.

        Args:
            years_ahead: Number of years to project.
            method: Projection method ('exponential', 'linear', 'logistic').

        Returns:
            List of PopulationProjection objects.
        """
        if 'population' not in self.data.columns:
            logger.error("Population column not found")
            return []

        df = self.data.dropna(subset=['population']).copy()
        df = df.sort_values('year')
        
        latest_year = int(df['year'].max())
        years = df['year'].values
        population = df['population'].values

        projections = []

        if method == 'exponential':
            # Fit exponential model using recent data
            recent_df = df[df['year'] >= 2000]
            if len(recent_df) >= 5:
                x = recent_df['year'].values - recent_df['year'].min()
                y = recent_df['population'].values
                
                # Log-linear regression
                log_y = np.log(y)
                slope, intercept, r_value, _, std_err = stats.linregress(
                    x, log_y
                )
                
                for i in range(1, years_ahead + 1):
                    future_year = latest_year + i
                    x_pred = future_year - recent_df['year'].min()
                    
                    predicted = np.exp(intercept + slope * x_pred)
                    # Confidence interval (approximate)
                    margin = predicted * (1.96 * std_err * i / 10)
                    
                    projections.append(PopulationProjection(
                        year=future_year,
                        projected_population=predicted,
                        lower_bound=predicted - margin,
                        upper_bound=predicted + margin,
                        model_type='exponential'
                    ))

        elif method == 'linear':
            slope, intercept, r_value, _, std_err = stats.linregress(
                years, population
            )
            
            for i in range(1, years_ahead + 1):
                future_year = latest_year + i
                predicted = intercept + slope * future_year
                margin = 1.96 * std_err * np.sqrt(
                    1 + 1/len(years) +
                    (future_year - np.mean(years))**2 /
                    np.sum((years - np.mean(years))**2)
                ) * np.std(population)
                
                projections.append(PopulationProjection(
                    year=future_year,
                    projected_population=max(0, predicted),
                    lower_bound=max(0, predicted - margin),
                    upper_bound=predicted + margin,
                    model_type='linear'
                ))

        elif method == 'logistic':
            try:
                # Estimate initial parameters
                k_init = population.max() * 1.5
                r_init = 0.02
                m_init = 2000
                
                popt, pcov = curve_fit(
                    self._logistic_growth,
                    years,
                    population,
                    p0=[k_init, r_init, m_init],
                    maxfev=10000,
                    bounds=([population.max(), 0, 1960], [500e6, 0.1, 2100])
                )
                
                perr = np.sqrt(np.diag(pcov))
                
                for i in range(1, years_ahead + 1):
                    future_year = latest_year + i
                    predicted = self._logistic_growth(future_year, *popt)
                    margin = predicted * 0.05 * i / 10
                    
                    projections.append(PopulationProjection(
                        year=future_year,
                        projected_population=predicted,
                        lower_bound=predicted - margin,
                        upper_bound=predicted + margin,
                        model_type='logistic'
                    ))
            except (RuntimeError, ValueError) as e:
                logger.warning(f"Logistic fit failed: {e}")
                return self.project_population(years_ahead, 'exponential')

        self.analysis_results[f'projections_{method}'] = [
            vars(p) for p in projections
        ]
        return projections

    def analyze_urbanization(self) -> Dict[str, any]:
        """
        Analyze urbanization trends and their impact.

        Returns:
            Dictionary with urbanization analysis.
        """
        if 'urban_population_pct' not in self.data.columns:
            logger.warning("Urban population data not available")
            return {}

        df = self.data.dropna(subset=['urban_population_pct']).copy()
        df = df.sort_values('year')

        if df.empty:
            return {}

        latest = df.iloc[-1]
        earliest = df.iloc[0]

        urbanization_analysis = {
            'current_urban_pct': round(latest['urban_population_pct'], 2),
            'earliest_urban_pct': round(earliest['urban_population_pct'], 2),
            'urbanization_increase': round(
                latest['urban_population_pct'] -
                earliest['urban_population_pct'],
                2
            ),
            'annual_urbanization_rate': round(
                (latest['urban_population_pct'] -
                 earliest['urban_population_pct']) /
                (latest['year'] - earliest['year']),
                3
            ),
            'implications': [
                "Increased pressure on urban infrastructure",
                "Growing demand for housing and services",
                "Environmental challenges in cities",
                "Rural-urban migration patterns",
            ],
        }

        self.analysis_results['urbanization'] = urbanization_analysis
        return urbanization_analysis

    def analyze_fertility_trends(self) -> Dict[str, any]:
        """
        Analyze fertility rate trends and demographic transition.

        Returns:
            Dictionary with fertility analysis.
        """
        if 'fertility_rate' not in self.data.columns:
            return {}

        df = self.data.dropna(subset=['fertility_rate']).copy()
        df = df.sort_values('year')

        if df.empty:
            return {}

        latest = df.iloc[-1]
        earliest = df.iloc[0]
        max_fertility = df.loc[df['fertility_rate'].idxmax()]

        fertility_analysis = {
            'current_fertility_rate': round(latest['fertility_rate'], 2),
            'peak_fertility_rate': round(max_fertility['fertility_rate'], 2),
            'peak_year': int(max_fertility['year']),
            'fertility_decline': round(
                max_fertility['fertility_rate'] - latest['fertility_rate'],
                2
            ),
            'replacement_level': 2.1,
            'years_to_replacement': self._estimate_years_to_replacement(df),
            'demographic_transition_stage': (
                self._determine_transition_stage(latest['fertility_rate'])
            ),
        }

        self.analysis_results['fertility'] = fertility_analysis
        return fertility_analysis

    def _estimate_years_to_replacement(
        self,
        df: pd.DataFrame,
        replacement_rate: float = 2.1
    ) -> Optional[int]:
        """Estimate years until replacement fertility level is reached."""
        recent = df[df['year'] >= 2010].copy()
        if len(recent) < 3:
            return None

        slope, intercept, _, _, _ = stats.linregress(
            recent['year'], recent['fertility_rate']
        )
        
        if slope >= 0:
            return None
        
        years_needed = (replacement_rate - intercept) / slope
        current_year = recent['year'].max()
        
        if years_needed > current_year:
            return int(years_needed - current_year)
        return 0

    @staticmethod
    def _determine_transition_stage(fertility_rate: float) -> str:
        """Determine demographic transition stage based on fertility."""
        if fertility_rate > 5:
            return "Stage 1-2: Pre-transition/Early transition"
        elif fertility_rate > 3:
            return "Stage 2-3: Mid-transition"
        elif fertility_rate > 2.1:
            return "Stage 3: Late transition"
        else:
            return "Stage 4: Post-transition"

    def generate_summary_report(self) -> str:
        """
        Generate a comprehensive text summary of all analyses.

        Returns:
            Formatted summary report string.
        """
        report_lines = [
            "\n" + "=" * 70,
            "BANGLADESH POPULATION ANALYSIS - SUMMARY REPORT",
            "=" * 70 + "\n",
        ]

        # Basic Statistics
        if 'population_statistics' in self.analysis_results:
            stats_data = self.analysis_results['population_statistics']
            report_lines.extend([
                "1. BASIC STATISTICS",
                "-" * 40,
                f"   Data points: {stats_data.get('count', 'N/A')}",
                f"   Mean population: {format_population(stats_data.get('mean', 0))}",
                f"   Min population: {format_population(stats_data.get('min', 0))}",
                f"   Max population: {format_population(stats_data.get('max', 0))}",
                "",
            ])

        # Growth Trends
        if 'growth_trends' in self.analysis_results:
            report_lines.extend([
                "2. GROWTH TRENDS",
                "-" * 40,
            ])
            for period, data in self.analysis_results['growth_trends'].items():
                report_lines.append(
                    f"   {period}: {data['annual_growth_rate']}% annual growth, "
                    f"doubling time: {data['doubling_time_years']} years"
                )
            report_lines.append("")

        # Density Analysis
        if 'density_analysis' in self.analysis_results:
            density = self.analysis_results['density_analysis']
            report_lines.extend([
                "3. POPULATION DENSITY",
                "-" * 40,
                f"   Current density: {density['current_density']} per km²",
                f"   Density increase: {density['density_increase_pct']}%",
                "",
            ])

        # Urbanization
        if 'urbanization' in self.analysis_results:
            urban = self.analysis_results['urbanization']
            report_lines.extend([
                "4. URBANIZATION",
                "-" * 40,
                f"   Current urban population: {urban['current_urban_pct']}%",
                f"   Urbanization increase: {urban['urbanization_increase']}%",
                "",
            ])

        # Fertility
        if 'fertility' in self.analysis_results:
            fertility = self.analysis_results['fertility']
            report_lines.extend([
                "5. FERTILITY TRENDS",
                "-" * 40,
                f"   Current fertility rate: {fertility['current_fertility_rate']}",
                f"   Peak fertility: {fertility['peak_fertility_rate']} "
                f"(Year: {fertility['peak_year']})",
                f"   Stage: {fertility['demographic_transition_stage']}",
                "",
            ])

        report_lines.append("=" * 70)
        
        return "\n".join(report_lines)


class DivisionAnalyzer:
    """
    Class for analyzing division-level population data.
    """

    def __init__(self, division_data: pd.DataFrame):
        """
        Initialize the DivisionAnalyzer.

        Args:
            division_data: DataFrame with division-level population data.
        """
        self.data = division_data.copy()
        logger.info("DivisionAnalyzer initialized")

    def calculate_density_rankings(self) -> pd.DataFrame:
        """
        Calculate and rank divisions by population density.

        Returns:
            DataFrame with density rankings.
        """
        df = self.data.copy()
        df['Density_Rank'] = df['Density_2022'].rank(ascending=False)
        df = df.sort_values('Density_Rank')
        return df[['Division', 'Density_2022', 'Density_Rank', 'Area_km2']]

    def identify_overpopulation_stress(self) -> Dict[str, any]:
        """
        Identify divisions under population stress.

        Returns:
            Dictionary with stress analysis results.
        """
        df = self.data.copy()
        
        # Define thresholds
        high_density_threshold = df['Density_2022'].mean() + df['Density_2022'].std()
        high_growth_threshold = df['Growth_Rate'].mean() + df['Growth_Rate'].std()
        
        stress_analysis = {
            'high_density_divisions': df[
                df['Density_2022'] > high_density_threshold
            ]['Division'].tolist(),
            'high_growth_divisions': df[
                df['Growth_Rate'] > high_growth_threshold
            ]['Division'].tolist(),
            'critical_divisions': df[
                (df['Density_2022'] > high_density_threshold) &
                (df['Growth_Rate'] > high_growth_threshold)
            ]['Division'].tolist(),
            'density_threshold': round(high_density_threshold, 2),
            'growth_threshold': round(high_growth_threshold, 3),
        }
        
        return stress_analysis