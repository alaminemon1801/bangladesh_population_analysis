"""
Visualization module for Bangladesh Population Analysis Project.

This module provides classes and functions for creating
various charts and graphs to visualize population data.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

from config import COLOR_PALETTE, FIGURE_SIZE, OUTPUT_DIR, PLOT_STYLE
from data_processor import PopulationProjection
from utils import format_population, logger


class PopulationVisualizer:
    """
    Class for creating population data visualizations.
    
    This class provides methods for generating various charts
    and plots related to population analysis.
    """

    def __init__(self, output_dir: Path = OUTPUT_DIR):
        """
        Initialize the PopulationVisualizer.

        Args:
            output_dir: Directory for saving output figures.
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style
        try:
            plt.style.use(PLOT_STYLE)
        except OSError:
            plt.style.use('seaborn-v0_8')
        
        self.colors = COLOR_PALETTE
        logger.info("PopulationVisualizer initialized")

    def plot_population_timeline(
        self,
        data: pd.DataFrame,
        save: bool = True,
        show: bool = True
    ) -> Optional[Path]:
        """
        Create a timeline plot of population growth.

        Args:
            data: DataFrame with 'year' and 'population' columns.
            save: Whether to save the figure.
            show: Whether to display the figure.

        Returns:
            Path to saved figure, or None if not saved.
        """
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        
        ax.plot(
            data['year'],
            data['population'] / 1e6,
            linewidth=2.5,
            color=self.colors[0],
            marker='o',
            markersize=4,
            label='Population'
        )
        
        ax.fill_between(
            data['year'],
            data['population'] / 1e6,
            alpha=0.3,
            color=self.colors[0]
        )
        
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Population (Millions)', fontsize=12)
        ax.set_title(
            'Bangladesh Population Growth (1960-2023)',
            fontsize=14,
            fontweight='bold'
        )
        
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, p: f'{x:.0f}M')
        )
        
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        
        # Add annotations for key milestones
        self._add_milestone_annotations(ax, data)
        
        plt.tight_layout()
        
        filepath = None
        if save:
            filepath = self.output_dir / 'population_timeline.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved timeline plot to {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return filepath

    def _add_milestone_annotations(
        self,
        ax: plt.Axes,
        data: pd.DataFrame
    ) -> None:
        """Add annotations for population milestones."""
        milestones = [
            (1971, "Independence"),
            (2000, "New Millennium"),
            (2022, "Latest Census"),
        ]
        
        for year, label in milestones:
            row = data[data['year'] == year]
            if not row.empty:
                pop = row['population'].values[0] / 1e6
                ax.annotate(
                    label,
                    xy=(year, pop),
                    xytext=(year + 3, pop + 10),
                    fontsize=9,
                    arrowprops=dict(arrowstyle='->', color='gray'),
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                )

    def plot_growth_rate_trend(
        self,
        data: pd.DataFrame,
        save: bool = True,
        show: bool = True
    ) -> Optional[Path]:
        """
        Plot population growth rate over time.

        Args:
            data: DataFrame with 'year' and 'growth_rate' columns.
            save: Whether to save the figure.
            show: Whether to display the figure.

        Returns:
            Path to saved figure, or None if not saved.
        """
        if 'growth_rate' not in data.columns:
            logger.warning("Growth rate data not available")
            return None

        df = data.dropna(subset=['growth_rate'])
        
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        
        # Plot growth rate
        ax.plot(
            df['year'],
            df['growth_rate'],
            linewidth=2.5,
            color=self.colors[1],
            marker='s',
            markersize=4
        )
        
        # Add horizontal line at replacement level growth
        ax.axhline(
            y=0,
            color='red',
            linestyle='--',
            linewidth=1.5,
            alpha=0.7,
            label='Zero Growth'
        )
        
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Annual Growth Rate (%)', fontsize=12)
        ax.set_title(
            'Bangladesh Population Growth Rate Decline',
            fontsize=14,
            fontweight='bold'
        )
        
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        filepath = None
        if save:
            filepath = self.output_dir / 'growth_rate_trend.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved growth rate plot to {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return filepath

    def plot_population_projection(
        self,
        historical_data: pd.DataFrame,
        projections: List[PopulationProjection],
        save: bool = True,
        show: bool = True
    ) -> Optional[Path]:
        """
        Plot historical data with future projections.

        Args:
            historical_data: DataFrame with historical population data.
            projections: List of PopulationProjection objects.
            save: Whether to save the figure.
            show: Whether to display the figure.

        Returns:
            Path to saved figure, or None if not saved.
        """
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        
        # Plot historical data
        ax.plot(
            historical_data['year'],
            historical_data['population'] / 1e6,
            linewidth=2.5,
            color=self.colors[0],
            label='Historical'
        )
        
        # Plot projections
        proj_years = [p.year for p in projections]
        proj_pop = [p.projected_population / 1e6 for p in projections]
        proj_lower = [p.lower_bound / 1e6 for p in projections]
        proj_upper = [p.upper_bound / 1e6 for p in projections]
        
        ax.plot(
            proj_years,
            proj_pop,
            linewidth=2.5,
            color=self.colors[2],
            linestyle='--',
            label='Projected'
        )
        
        ax.fill_between(
            proj_years,
            proj_lower,
            proj_upper,
            alpha=0.2,
            color=self.colors[2],
            label='Confidence Interval'
        )
        
        # Mark transition point
        latest_hist_year = historical_data['year'].max()
        ax.axvline(
            x=latest_hist_year,
            color='gray',
            linestyle=':',
            linewidth=1.5,
            alpha=0.7
        )
        
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Population (Millions)', fontsize=12)
        ax.set_title(
            'Bangladesh Population Projection',
            fontsize=14,
            fontweight='bold'
        )
        
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = None
        if save:
            filepath = self.output_dir / 'population_projection.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved projection plot to {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return filepath

    def plot_division_comparison(
        self,
        division_data: pd.DataFrame,
        save: bool = True,
        show: bool = True
    ) -> Optional[Path]:
        """
        Create comparison charts for division-level data.

        Args:
            division_data: DataFrame with division-level population data.
            save: Whether to save the figure.
            show: Whether to display the figure.

        Returns:
            Path to saved figure, or None if not saved.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        df = division_data.sort_values('Population_2022', ascending=True)
        
        # Plot 1: Population by Division (Horizontal Bar)
        ax1 = axes[0, 0]
        bars1 = ax1.barh(
            df['Division'],
            df['Population_2022'] / 1e6,
            color=self.colors[:len(df)]
        )
        ax1.set_xlabel('Population (Millions)')
        ax1.set_title('Population by Division (2022)', fontweight='bold')
        ax1.bar_label(bars1, fmt='%.1fM', padding=3)
        
        # Plot 2: Population Density by Division
        ax2 = axes[0, 1]
        df_density = df.sort_values('Density_2022', ascending=True)
        bars2 = ax2.barh(
            df_density['Division'],
            df_density['Density_2022'],
            color=self.colors[:len(df)]
        )
        ax2.set_xlabel('Population Density (per km²)')
        ax2.set_title('Population Density by Division', fontweight='bold')
        ax2.bar_label(bars2, fmt='%.0f', padding=3)
        
        # Plot 3: Growth Rate by Division
        ax3 = axes[1, 0]
        df_growth = df.sort_values('Growth_Rate', ascending=True)
        colors_growth = [
            'red' if x > df['Growth_Rate'].mean() else self.colors[2]
            for x in df_growth['Growth_Rate']
        ]
        bars3 = ax3.barh(
            df_growth['Division'],
            df_growth['Growth_Rate'],
            color=colors_growth
        )
        ax3.set_xlabel('Annual Growth Rate (%)')
        ax3.set_title('Population Growth Rate by Division', fontweight='bold')
        ax3.bar_label(bars3, fmt='%.2f%%', padding=3)
        ax3.axvline(
            x=df['Growth_Rate'].mean(),
            color='black',
            linestyle='--',
            label='Average'
        )
        ax3.legend()
        
        # Plot 4: Pie Chart of Population Distribution
        ax4 = axes[1, 1]
        explode = [0.05 if d == 'Dhaka' else 0 for d in df['Division']]
        ax4.pie(
            df['Population_2022'],
            labels=df['Division'],
            autopct='%1.1f%%',
            colors=self.colors[:len(df)],
            explode=explode,
            startangle=90
        )
        ax4.set_title(
            'Population Distribution by Division',
            fontweight='bold'
        )
        
        plt.tight_layout()
        
        filepath = None
        if save:
            filepath = self.output_dir / 'division_comparison.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved division comparison plot to {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return filepath

    def plot_demographic_indicators(
        self,
        data: pd.DataFrame,
        save: bool = True,
        show: bool = True
    ) -> Optional[Path]:
        """
        Plot multiple demographic indicators together.

        Args:
            data: DataFrame with multiple indicator columns.
            save: Whether to save the figure.
            show: Whether to display the figure.

        Returns:
            Path to saved figure, or None if not saved.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Fertility Rate
        if 'fertility_rate' in data.columns:
            ax1 = axes[0, 0]
            df_fert = data.dropna(subset=['fertility_rate'])
            ax1.plot(
                df_fert['year'],
                df_fert['fertility_rate'],
                linewidth=2,
                color=self.colors[3],
                marker='o',
                markersize=3
            )
            ax1.axhline(
                y=2.1,
                color='red',
                linestyle='--',
                label='Replacement Level'
            )
            ax1.set_xlabel('Year')
            ax1.set_ylabel('Fertility Rate')
            ax1.set_title('Fertility Rate Trend', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # Plot 2: Birth and Death Rates
        ax2 = axes[0, 1]
        if 'birth_rate' in data.columns:
            df_birth = data.dropna(subset=['birth_rate'])
            ax2.plot(
                df_birth['year'],
                df_birth['birth_rate'],
                linewidth=2,
                color=self.colors[0],
                label='Birth Rate'
            )
        if 'death_rate' in data.columns:
            df_death = data.dropna(subset=['death_rate'])
            ax2.plot(
                df_death['year'],
                df_death['death_rate'],
                linewidth=2,
                color=self.colors[1],
                label='Death Rate'
            )
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Rate (per 1,000)')
        ax2.set_title('Birth and Death Rates', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Urban Population Percentage
        if 'urban_population_pct' in data.columns:
            ax3 = axes[1, 0]
            df_urban = data.dropna(subset=['urban_population_pct'])
            ax3.fill_between(
                df_urban['year'],
                df_urban['urban_population_pct'],
                alpha=0.4,
                color=self.colors[4]
            )
            ax3.plot(
                df_urban['year'],
                df_urban['urban_population_pct'],
                linewidth=2,
                color=self.colors[4]
            )
            ax3.set_xlabel('Year')
            ax3.set_ylabel('Urban Population (%)')
            ax3.set_title('Urbanization Trend', fontweight='bold')
            ax3.grid(True, alpha=0.3)

        # Plot 4: Population Density over time
        if 'density' in data.columns:
            ax4 = axes[1, 1]
            df_dens = data.dropna(subset=['density'])
            ax4.plot(
                df_dens['year'],
                df_dens['density'],
                linewidth=2,
                color=self.colors[5],
                marker='s',
                markersize=3
            )
            ax4.set_xlabel('Year')
            ax4.set_ylabel('Population Density (per km²)')
            ax4.set_title('Population Density Trend', fontweight='bold')
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        
        filepath = None
        if save:
            filepath = self.output_dir / 'demographic_indicators.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved demographic indicators plot to {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return filepath

    def create_infographic_summary(
        self,
        analysis_results: Dict,
        save: bool = True,
        show: bool = True
    ) -> Optional[Path]:
        """
        Create an infographic-style summary visualization.

        Args:
            analysis_results: Dictionary containing analysis results.
            save: Whether to save the figure.
            show: Whether to display the figure.

        Returns:
            Path to saved figure, or None if not saved.
        """
        fig = plt.figure(figsize=(16, 10))
        
        # Title
        fig.suptitle(
            'BANGLADESH POPULATION: KEY INSIGHTS',
            fontsize=20,
            fontweight='bold',
            y=0.98
        )
        
        # Create grid
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Key Statistics Boxes
        stats_data = [
            ("Current Population", "170+ Million", self.colors[0]),
            ("Population Density", "1,265/km²", self.colors[1]),
            ("Growth Rate", "1.0%", self.colors[2]),
            ("Urban Population", "39%", self.colors[3]),
        ]
        
        for idx, (title, value, color) in enumerate(stats_data):
            ax = fig.add_subplot(gs[0, idx])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            
            # Background rectangle
            rect = plt.Rectangle(
                (0.05, 0.05), 0.9, 0.9,
                facecolor=color,
                alpha=0.2,
                edgecolor=color,
                linewidth=2
            )
            ax.add_patch(rect)
            
            ax.text(
                0.5, 0.65, value,
                ha='center', va='center',
                fontsize=18, fontweight='bold',
                color=color
            )
            ax.text(
                0.5, 0.3, title,
                ha='center', va='center',
                fontsize=10
            )
            ax.axis('off')
        
        # Key Findings Text
        ax_findings = fig.add_subplot(gs[1:, :2])
        findings_text = """
KEY FINDINGS ON OVERPOPULATION:

1. DENSITY CRISIS
   • Bangladesh ranks among the world's most 
     densely populated countries
   • Dhaka division alone houses ~26% of population

2. DECLINING GROWTH
   • Growth rate decreased from 3% (1970s) to 1% (2020s)
   • Fertility rate dropped from 6.9 to 2.0

3. URBANIZATION PRESSURE
   • Urban population growing at 3% annually
   • Major cities face infrastructure strain

4. FUTURE OUTLOOK
   • Population expected to stabilize around 200M
   • Demographic dividend opportunity exists

5. POLICY IMPLICATIONS
   • Continued family planning success
   • Need for sustainable urban development
   • Focus on education and healthcare
        """
        ax_findings.text(
            0.05, 0.95, findings_text,
            transform=ax_findings.transAxes,
            fontsize=10,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        )
        ax_findings.axis('off')
        
        # Timeline mini-chart
        ax_timeline = fig.add_subplot(gs[1, 2:])
        years = [1960, 1980, 2000, 2020, 2050]
        pops = [50, 80, 130, 170, 195]
        ax_timeline.plot(
            years, pops,
            marker='o', linewidth=2,
            color=self.colors[0],
            markersize=10
        )
        ax_timeline.fill_between(
            years, pops,
            alpha=0.2,
            color=self.colors[0]
        )
        ax_timeline.set_title(
            'Population Milestones (Millions)',
            fontweight='bold'
        )
        ax_timeline.set_xlabel('Year')
        ax_timeline.grid(True, alpha=0.3)
        
        # Recommendations Box
        ax_rec = fig.add_subplot(gs[2, 2:])
        rec_text = """
RECOMMENDATIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Invest in education, especially for women
✓ Strengthen healthcare infrastructure  
✓ Promote sustainable urbanization
✓ Support rural development programs
✓ Continue successful family planning initiatives
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
        ax_rec.text(
            0.5, 0.5, rec_text,
            transform=ax_rec.transAxes,
            fontsize=10,
            verticalalignment='center',
            horizontalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3)
        )
        ax_rec.axis('off')
        
        filepath = None
        if save:
            filepath = self.output_dir / 'infographic_summary.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved infographic to {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return filepath