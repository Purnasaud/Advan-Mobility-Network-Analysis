"""
Configuration file for Wyoming Mobility Analysis
Author: Purna Saud
Date: 2025

This module contains all configuration parameters, paths, and styling settings.
"""

from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Path Configuration

BASE_DIR = Path(r"F:\Yellow Stone Flood-2022\floodanalysis\Data\Advan Mobility")
RAW_DATA_FILE = BASE_DIR / "Advan_2018_2024_US_WY.csv"
PROCESSED_DATA_FILE = BASE_DIR / "WY_Monthly_Mobility_Metrics_2018-2024.xlsx"
OUTPUT_DIR = BASE_DIR / "Mobility_Metrics_Plots"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Wyoing counties data
WYOMING_COUNTY_NAMES = {
    "56001": "Albany", "56003": "Big Horn", "56005": "Campbell", "56007": "Carbon",
    "56009": "Converse", "56011": "Crook", "56013": "Fremont", "56015": "Goshen",
    "56017": "Hot Springs", "56019": "Johnson", "56021": "Laramie", "56023": "Lincoln",
    "56025": "Natrona", "56027": "Niobrara", "56029": "Park", "56031": "Platte",
    "56033": "Sheridan", "56035": "Sublette", "56037": "Sweetwater", "56039": "Teton",
    "56041": "Uinta", "56043": "Washakie", "56045": "Weston"
}

WY_COUNTY_FIPS = set(WYOMING_COUNTY_NAMES.keys())

# Time period configuration
MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
}
MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
YEAR_RANGE = (2018, 2024)

# COVID-19 period definition
COVID_PERIODS = {
    'pre': ['2018', '2019'],
    'during': ['2020', '2021', '2022', '2023'],
    'post': ['2024']
}

# Metrics configuration and all available metrics
ALL_METRICS = [
    'Total_Inflow', 'Internal_Flow', 'Inflow_InState', 'Inflow_OutState',
    'Outflow_to_WY', 'Distinct_Origins_Total', 'Distinct_Origins_WY',
    'Distinct_Origins_OutState', 'InDegree', 'OutDegree', 'Weighted_InDegree',
    'Weighted_OutDegree', 'Weighted_InStrength', 'Weighted_OutStrength',
    'Closeness', 'Weighted_Closeness', 'Betweenness', 'Eigenvector', 'PageRank'
]

# Key metrics for standard analysis
KEY_METRICS = [
    'InDegree', 'OutDegree', 'Weighted_InStrength', 'Weighted_OutStrength',
    'PageRank', 'Betweenness', 'Closeness', 'Eigenvector'
]

# Metrics for correlation analysis
CORRELATION_METRICS = [
    'InDegree', 'PageRank', 'Betweenness', 'Eigenvector',
    'Weighted_InStrength', 'Closeness'
]

# Visualization global matplotlib setting
PLOT_STYLE = {
    "font.family": "Arial",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
}

# Color palettes
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'covid': '#E63946',
    'pre_covid': '#457B9D',
    'during_covid': '#E76F51',
    'post_covid': '#06A77D',
}

# Plot dimensions
FIGURE_SIZES = {
    'small': (7, 4),
    'medium': (10, 6),
    'large': (14, 8),
    'wide': (12, 4),
    'tall': (8, 10),
}

# Helper Functions
def setup_plot_style():
    """Apply global matplotlib style settings."""
    mpl.rcParams.update(PLOT_STYLE)

def format_period_label(period_str, format_type='short'):
    """
    Format time period labels for plots.
    
    Parameters:
    -----------
    period_str : str
        Period string like '2020-Jul'
    format_type : str
    Returns:
    --------
    str : Formatted period label
    """
    try:
        year, month = period_str.split('-')
        
        if format_type == 'short':
            return f"{month[:3]} '{year[-2:]}"
        elif format_type == 'medium':
            return f"{month[:3]} {year}"
        elif format_type == 'long':
            month_full = {
                'Jan': 'January', 'Feb': 'February', 'Mar': 'March',
                'Apr': 'April', 'May': 'May', 'Jun': 'June',
                'Jul': 'July', 'Aug': 'August', 'Sep': 'September',
                'Oct': 'October', 'Nov': 'November', 'Dec': 'December'
            }
            return f"{month_full.get(month, month)} {year}"
        else:
            return period_str
    except:
        return period_str


def get_year_boundaries(time_periods):
    """
    Get indices where years change for vertical lines in plots.
    
    Parameters:
    -----------
    time_periods : list
        List of period strings like ['2018-Jan', '2018-Feb', ...]
    
    Returns:
    --------
    dict : Dictionary with years as keys and indices as values
    """
    boundaries = {}
    current_year = None
    
    for idx, period in enumerate(time_periods):
        year = period.split('-')[0]
        if year != current_year:
            boundaries[year] = idx
            current_year = year
    
    return boundaries


def find_covid_index(time_periods):
    """
    Find the index where COVID-19 period starts (2020).
    Parameters:
    -----------
    time_periods : list
        List of period strings
    
    Returns:
    --------
    int or None : Index of first 2020 period
    """
    for i, period in enumerate(time_periods):
        if "2020" in period:
            return i
    return None

def is_covid_period(period_str):
    """
    Check if a period falls within COVID-19 time frame.
    Parameters:
    -----------
    period_str : str
        Period string like '2020-Jul'
    Returns:
    --------
    bool : True if period is during COVID (2020-2023)
    """
    year = period_str.split('-')[0]
    return year in COVID_PERIODS['during']


def get_county_colors(counties, colormap='tab20'):
    """
    Generate consistent color mapping for counties.
    Parameters:
    -----------
    counties : list
        List of county names
    colormap : str
        Name of matplotlib colormap
    Returns:
    --------
    dict : Dictionary mapping county names to colors
    """
    cmap = plt.get_cmap(colormap)
    n_colors = cmap.N
    return {county: cmap(i % n_colors) for i, county in enumerate(sorted(counties))}


def create_year_tick_labels(time_periods, tick_frequency='yearly'):
    """
    Create clean year-based tick labels and positions.
    Parameters:
    -----------
    time_periods : list
        List of period strings
    tick_frequency : str
        'yearly' - one tick per year (Jan of each year)
        'quarterly' - one tick per quarter
        'monthly' - all months
    Returns:
    --------
    tuple : (tick_positions, tick_labels)
    """
    positions = []
    labels = []
    
    if tick_frequency == 'yearly':
        # One tick per year at January
        for idx, period in enumerate(time_periods):
            year, month = period.split('-')
            if month == 'Jan':
                positions.append(idx)
                labels.append(year)
    
    elif tick_frequency == 'quarterly':
        # Quarterly ticks (Jan, Apr, Jul, Oct)
        for idx, period in enumerate(time_periods):
            year, month = period.split('-')
            if month in ['Jan', 'Apr', 'Jul', 'Oct']:
                positions.append(idx)
                labels.append(format_period_label(period, 'short'))
    
    else:  # monthly
        positions = list(range(len(time_periods)))
        labels = [format_period_label(p, 'short') for p in time_periods]
    
    return positions, labels


def get_metric_display_name(metric_key):
    """
    Convert metric key to clean display name.
    Parameters:
    -----------
    metric_key : str
        Metric key like 'Weighted_InStrength'
    
    Returns:
    --------
    str : Display name like 'Weighted In-Strength'
    """
    # Replace underscores with spaces
    name = metric_key.replace('_', ' ')
    name = name.replace('InStrength', 'In-Strength')
    name = name.replace('OutStrength', 'Out-Strength')
    name = name.replace('InDegree', 'In-Degree')
    name = name.replace('OutDegree', 'Out-Degree')
    name = name.replace('InState', 'In-State')
    name = name.replace('OutState', 'Out-of-State')
    name = name.replace('PageRank', 'PageRank')
    
    return name

# Validation Functions
def validate_paths():
    """Validate that all required paths exist."""
    if not BASE_DIR.exists():
        raise FileNotFoundError(f"Base directory not found: {BASE_DIR}")
    
    if not PROCESSED_DATA_FILE.exists():
        print(f"Warning: Processed data file not found: {PROCESSED_DATA_FILE}")
        return False
    
    print("All paths validated")
    return True


def print_config_summary():
    """Print configuration summary."""
    print("CONFIGURATION SUMMARY")
    print("="*70)
    print(f"Base Directory:     {BASE_DIR}")
    print(f"Output Directory:   {OUTPUT_DIR}")
    print(f"Year Range:         {YEAR_RANGE[0]} - {YEAR_RANGE[1]}")
    print(f"Number of Counties: {len(WYOMING_COUNTY_NAMES)}")
    print(f"Total Metrics:      {len(ALL_METRICS)}")
    print(f"Key Metrics:        {len(KEY_METRICS)}")

# initialization
if __name__ == "__main__":
    setup_plot_style()
    print_config_summary()
    validate_paths()