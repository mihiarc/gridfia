"""
Vance County NDVI Analysis Runner

This script orchestrates the complete NDVI analysis workflow for Vance County,
including statistical analysis and visualization generation. It processes the
data prepared by the data_processing module and generates comprehensive
analysis outputs.
"""
import logging
import sys
import argparse
from pathlib import Path

from .config.analysis_config import (
    PROCESSED_NDVI,
    OUTPUT_DIR,
    STATS_OUTPUT,
    TREND_PLOT,
    DISTRIBUTION_PLOT,
    SUMMARY_REPORT,
    LOG_FORMAT,
    LOG_LEVEL,
    LOG_FILE
)
from .stats.stats_analyzer import NDVIStatsAnalyzer
from .visualization.ndvi_plotter import NDVIPlotter

def setup_logging():
    """Configure logging for the analysis pipeline."""
    log_file = OUTPUT_DIR / LOG_FILE
    
    # Create handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Create formatters and add it to handlers
    formatter = logging.Formatter(LOG_FORMAT)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Log initial information
    logger = logging.getLogger(__name__)
    logger.info("Starting Vance County NDVI Analysis")
    logger.info(f"Log file: {log_file}")
    
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run Vance County NDVI analysis pipeline'
    )
    parser.add_argument(
        '--input-file',
        type=Path,
        default=PROCESSED_NDVI,
        help='Path to processed NDVI data'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=OUTPUT_DIR,
        help='Directory for analysis outputs'
    )
    parser.add_argument(
        '--skip-plots',
        action='store_true',
        help='Skip plot generation'
    )
    return parser.parse_args()
    
def main():
    """Run the complete Vance County NDVI analysis pipeline."""
    # Setup logging first
    setup_logging()
    logger = logging.getLogger(__name__)
    
    args = parse_args()
    
    try:
        # Ensure input file exists
        if not args.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {args.input_file}")
            
        # Create output directory
        args.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Statistical Analysis
        logger.info("Starting statistical analysis...")
        stats_analyzer = NDVIStatsAnalyzer()
        stats_analyzer.load_data(args.input_file)
        
        # Calculate and save basic statistics
        basic_stats = stats_analyzer.calculate_basic_statistics()
        for group, stats in basic_stats.items():
            output_file = args.output_dir / f"{group}_yearly_stats.csv"
            stats.to_csv(output_file)
            logger.info(f"Saved {group} yearly statistics to {output_file}")
            
        # Calculate and save trend statistics
        trend_stats, trend_tests = stats_analyzer.calculate_trend_statistics()
        trend_stats.to_csv(STATS_OUTPUT)
        logger.info(f"Saved trend statistics to {STATS_OUTPUT}")
        
        # Calculate yearly differences
        yearly_diff = stats_analyzer.calculate_yearly_differences()
        yearly_diff.to_csv(args.output_dir / "yearly_differences.csv")
        logger.info("Saved yearly differences analysis")
        
        # Generate summary report
        report = stats_analyzer.generate_summary_report()
        SUMMARY_REPORT.write_text(report)
        logger.info(f"Saved analysis summary to {SUMMARY_REPORT}")
        
        # Step 2: Visualization (if not skipped)
        if not args.skip_plots:
            logger.info("Starting visualization generation...")
            plotter = NDVIPlotter(stats_analyzer.data)
            
            # Create all plots
            plot_paths = plotter.create_all_plots(args.output_dir)
            for plot_name, path in plot_paths.items():
                logger.info(f"Generated {plot_name} plot: {path}")
                
        # Log Analysis Summary
        logger.info("\nAnalysis Summary:")
        logger.info(f"Total properties analyzed: {len(stats_analyzer.data)}")
        logger.info(f"Heirs properties: {stats_analyzer.data['is_heir'].sum()}")
        logger.info(f"Non-heirs properties: {(~stats_analyzer.data['is_heir']).sum()}")
        
        if trend_tests['significant']:
            logger.info("\nSignificant trend difference found:")
            logger.info(f"- T-statistic: {trend_tests['t_statistic']:.4f}")
            logger.info(f"- P-value: {trend_tests['p_value']:.4f}")
            logger.info(f"- Effect size (Cohen's d): {trend_tests['cohens_d']:.4f}")
        else:
            logger.info("\nNo significant trend difference found")
            
        logger.info("\nAnalysis outputs saved to:")
        logger.info(f"- Statistics: {args.output_dir}")
        logger.info(f"- Summary report: {SUMMARY_REPORT}")
        if not args.skip_plots:
            logger.info(f"- Visualizations: {args.output_dir}")
            
        logger.info("\nAnalysis completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return 1
        
if __name__ == '__main__':
    sys.exit(main()) 