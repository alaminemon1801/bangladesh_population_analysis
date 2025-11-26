#!/usr/bin/env python3
"""
Bangladesh Population Analysis - Main Application

This is the main entry point for the Bangladesh Population Analysis project.
It orchestrates data loading, processing, analysis, and visualization.

Author: Alamin Ahmed Emon

Data Sources:
- World Bank Open Data (https://data.worldbank.org/)
- Bangladesh Bureau of Statistics (https://bbs.gov.bd/)
"""

import sys
from datetime import datetime
from pathlib import Path

from config import DATA_DIR, OUTPUT_DIR
from data_loader import DataLoader
from data_processor import DivisionAnalyzer, PopulationAnalyzer
from utils import generate_report_header, logger, save_json
from visualizer import PopulationVisualizer


def main():
    """
    Main function to run the Bangladesh Population Analysis.
    
    This function orchestrates the entire analysis workflow:
    1. Data loading from APIs and local sources
    2. Data processing and analysis
    3. Visualization generation
    4. Report generation
    """
    print(generate_report_header())
    
    # =========================================================================
    # STEP 1: DATA LOADING
    # =========================================================================
    logger.info("=" * 50)
    logger.info("STEP 1: Loading Data")
    logger.info("=" * 50)
    
    # Initialize data loader
    loader = DataLoader(cache_enabled=True)
    
    # Load all indicators from World Bank API
    print("\nFetching data from World Bank API...")
    indicators = loader.load_all_indicators()
    
    if not indicators:
        logger.error("Failed to load data. Exiting.")
        sys.exit(1)
    
    # Create combined dataset
    combined_data = loader.create_combined_dataset(indicators)
    print(f"Loaded data for years: {combined_data['year'].min()} - "
          f"{combined_data['year'].max()}")
    print(f"Total data points: {len(combined_data)}")
    
    # Load division-level data
    division_data = loader.load_division_population_data()
    print(f"Loaded data for {len(division_data)} divisions")
    
    # Export raw data
    loader.export_data(combined_data, 'bangladesh_population_data', 'csv')
    loader.export_data(division_data, 'bangladesh_division_data', 'csv')
    
    # =========================================================================
    # STEP 2: DATA ANALYSIS
    # =========================================================================
    logger.info("=" * 50)
    logger.info("STEP 2: Analyzing Data")
    logger.info("=" * 50)
    
    # Initialize analyzer
    analyzer = PopulationAnalyzer(combined_data)
    
    # Run analyses
    print("\n--- Basic Statistics ---")
    stats = analyzer.calculate_basic_statistics('population')
    print(f"Population Range: {stats['min']:,.0f} - {stats['max']:,.0f}")
    print(f"Mean Population: {stats['mean']:,.0f}")
    
    print("\n--- Growth Trends ---")
    growth = analyzer.analyze_growth_trends()
    for period, data in growth.items():
        print(f"{period}: {data['annual_growth_rate']:.2f}% annual growth")
    
    print("\n--- Density Analysis ---")
    density = analyzer.calculate_density_analysis()
    print(f"Current Density: {density['current_density']:,.2f} per km²")
    print(f"Density Increase: {density['density_increase_pct']:.1f}%")
    
    print("\n--- Urbanization Analysis ---")
    urban = analyzer.analyze_urbanization()
    if urban:
        print(f"Current Urban Population: {urban['current_urban_pct']:.1f}%")
        print(f"Urbanization Rate: {urban['annual_urbanization_rate']:.2f}%/year")
    
    print("\n--- Fertility Analysis ---")
    fertility = analyzer.analyze_fertility_trends()
    if fertility:
        print(f"Current Fertility Rate: {fertility['current_fertility_rate']:.2f}")
        print(f"Peak Fertility: {fertility['peak_fertility_rate']:.2f} "
              f"(Year: {fertility['peak_year']})")
        print(f"Stage: {fertility['demographic_transition_stage']}")
    
    print("\n--- Population Projections ---")
    projections_exp = analyzer.project_population(30, 'exponential')
    if projections_exp:
        print("Exponential Model Projections:")
        for p in projections_exp[::10]:  # Print every 10th year
            print(f"  {p.year}: {p.projected_population/1e6:.1f} million "
                  f"({p.lower_bound/1e6:.1f} - {p.upper_bound/1e6:.1f})")
    
    # Division-level analysis
    print("\n--- Division Analysis ---")
    div_analyzer = DivisionAnalyzer(division_data)
    density_rankings = div_analyzer.calculate_density_rankings()
    print("\nPopulation Density Rankings:")
    print(density_rankings.to_string(index=False))
    
    stress_analysis = div_analyzer.identify_overpopulation_stress()
    print(f"\nHigh Density Divisions: {stress_analysis['high_density_divisions']}")
    print(f"High Growth Divisions: {stress_analysis['high_growth_divisions']}")
    
    # =========================================================================
    # STEP 3: VISUALIZATION
    # =========================================================================
    logger.info("=" * 50)
    logger.info("STEP 3: Creating Visualizations")
    logger.info("=" * 50)
    
    # Initialize visualizer
    visualizer = PopulationVisualizer(OUTPUT_DIR)
    
    print("\nGenerating visualizations...")
    
    # Create all plots
    pop_data = combined_data.dropna(subset=['population'])
    
    visualizer.plot_population_timeline(pop_data, save=True, show=False)
    print("✓ Population timeline created")
    
    visualizer.plot_growth_rate_trend(combined_data, save=True, show=False)
    print("✓ Growth rate trend created")
    
    visualizer.plot_population_projection(
        pop_data,
        projections_exp,
        save=True,
        show=False
    )
    print("✓ Population projection created")
    
    visualizer.plot_division_comparison(division_data, save=True, show=False)
    print("✓ Division comparison created")
    
    visualizer.plot_demographic_indicators(combined_data, save=True, show=False)
    print("✓ Demographic indicators created")
    
    visualizer.create_infographic_summary(
        analyzer.analysis_results,
        save=True,
        show=False
    )
    print("✓ Infographic summary created")
    
    # =========================================================================
    # STEP 4: GENERATE REPORT
    # =========================================================================
    logger.info("=" * 50)
    logger.info("STEP 4: Generating Report")
    logger.info("=" * 50)
    
    # Generate summary report
    summary_report = analyzer.generate_summary_report()
    print(summary_report)
    
    # Save analysis results to JSON
    save_json(analyzer.analysis_results, OUTPUT_DIR / 'analysis_results.json')
    print(f"\n✓ Analysis results saved to {OUTPUT_DIR / 'analysis_results.json'}")
    
    # =========================================================================
    # COMPLETION
    # =========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutput files saved to: {OUTPUT_DIR}")
    print(f"Data files saved to: {DATA_DIR}")
    print("\nGenerated files:")
    
    for f in OUTPUT_DIR.iterdir():
        print(f"  - {f.name}")
    
    print("\n" + "=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())