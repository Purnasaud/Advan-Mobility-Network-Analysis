"""
Visualization Functions for Wyoming Mobility Analysis
Professional publication-ready plots with clean formatting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
warnings.filterwarnings('ignore')

from advan_mobility_config import (
    PROCESSED_DATA_FILE, OUTPUT_DIR, format_period_label,
    get_county_colors, find_covid_index, is_covid_period,
    create_year_tick_labels, get_metric_display_name, COLORS
)

# Global variables (set by load_data)
data = {}
time_periods = []
time_numeric = []
counties = []
county_colors = {}
covid_idx = None

def load_data():
    """Load processed metrics data and initialize global variables"""
    global data, time_periods, time_numeric, counties, county_colors, covid_idx
    excel_file = pd.ExcelFile(PROCESSED_DATA_FILE)
    
    data = {sheet: pd.read_excel(PROCESSED_DATA_FILE, sheet_name=sheet, index_col=0)
            for sheet in excel_file.sheet_names}
    
    time_periods = data[excel_file.sheet_names[0]].columns.tolist()
    time_numeric = np.arange(len(time_periods))
    counties = data[excel_file.sheet_names[0]].index.tolist()
    county_colors = get_county_colors(counties)
    covid_idx = find_covid_index(time_periods)

    return data, time_periods, counties, covid_idx

# Time Series Plots
def plot_timeseries_top(metric_name, df=None, top_n=10, savepath=None):
    """
    Clean line plot showing top N counties by average metric value.
    
    Parameters:
    -----------
    metric_name : str
        Name of metric to plot
    df : pd.DataFrame, optional
        DataFrame for metric. If None, uses data[metric_name]
    top_n : int
        Number of top counties to show
    savepath : str or Path, optional
        Path to save figure
    """
    if df is None:
        df = data[metric_name]
    
    # Get top counties by mean
    mean_values = df.mean(axis=1).sort_values(ascending=False)
    top_counties = mean_values.head(top_n).index.tolist()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot lines
    for county in top_counties:
        y = df.loc[county].values
        ax.plot(time_numeric, y, marker='o', markersize=4, linewidth=2,
                color=county_colors[county], alpha=0.9, label=county)
    
    # COVID marker
    if covid_idx is not None:
        ax.axvline(x=covid_idx, color=COLORS['covid'], linestyle='--', 
                   linewidth=2, alpha=0.7, zorder=0)
        ax.text(covid_idx, ax.get_ylim()[1] * 0.98, "COVID-19",
                fontsize=10, color=COLORS['covid'], ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                         edgecolor=COLORS['covid'], alpha=0.8))
    
    # Year boundaries
    year_bounds = {}
    for idx, period in enumerate(time_periods):
        year = period.split('-')[0]
        if year not in year_bounds:
            year_bounds[year] = idx
    
    for year, idx in list(year_bounds.items())[1:]:  # Skip first
        ax.axvline(x=idx-0.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.3, zorder=0)
    
    # Labels and formatting
    ax.set_title(f"{get_metric_display_name(metric_name)} (Top {top_n} Counties)", 
                 weight="bold", fontsize=14, pad=15)
    ax.set_ylabel(get_metric_display_name(metric_name), fontsize=12, weight='bold')
    ax.set_xlabel("Time Period", fontsize=12, weight='bold')
    
    # Clean x-axis: show one tick per year
    tick_pos, tick_labs = create_year_tick_labels(time_periods, 'yearly')
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labs, fontsize=11)
    
    ax.grid(axis='y', alpha=0.3, linewidth=0.8)
    ax.set_xlim(-0.5, time_numeric[-1] + 0.5)
    
    # Legend outside plot area
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True,
              fontsize=10, title="County", title_fontsize=11, shadow=True)
    
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {savepath.name}")
    plt.show()


def plot_timeseries_all(metric_name, df=None, savepath=None):
    """
    Line plot showing ALL counties (more transparent).
    
    Parameters:
    -----------
    metric_name : str
        Name of metric to plot
    df : pd.DataFrame, optional
        DataFrame for metric
    savepath : str or Path, optional
        Path to save figure
    """
    if df is None:
        df = data[metric_name]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot all counties with transparency
    for county in df.index:
        y = df.loc[county].values
        ax.plot(time_numeric, y, linewidth=1.5, marker='o', markersize=3,
                color=county_colors[county], alpha=0.6, label=county)
    
    # COVID marker
    if covid_idx is not None:
        ax.axvline(x=covid_idx, color=COLORS['covid'], linestyle='--',
                   linewidth=2, alpha=0.7, zorder=0)
        ax.text(covid_idx, ax.get_ylim()[1] * 0.98, "COVID-19",
                fontsize=10, color=COLORS['covid'], ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                         edgecolor=COLORS['covid'], alpha=0.8))
    
    # Year boundaries
    year_bounds = {}
    for idx, period in enumerate(time_periods):
        year = period.split('-')[0]
        if year not in year_bounds:
            year_bounds[year] = idx
    
    for year, idx in list(year_bounds.items())[1:]:
        ax.axvline(x=idx-0.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.3, zorder=0)
    
    # Labels
    ax.set_title(f"{get_metric_display_name(metric_name)} (All {len(df.index)} Counties)",
                 weight="bold", fontsize=14, pad=15)
    ax.set_ylabel(get_metric_display_name(metric_name), fontsize=12, weight='bold')
    ax.set_xlabel("Time Period", fontsize=12, weight='bold')
    
    # Clean x-axis
    tick_pos, tick_labs = create_year_tick_labels(time_periods, 'yearly')
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labs, fontsize=11)
    
    ax.grid(axis='y', alpha=0.3, linewidth=0.8)
    
    # Legend in two columns
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True,
              fontsize=8, ncol=2, title="County", title_fontsize=9, shadow=True)
    
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {savepath.name}")
    plt.show()
    # plt.close(fig)

# Heat Map
def plot_heatmap(metric_name, df=None, top_n=15, savepath=None):
    """
    Heatmap showing metric values across time for top counties.
    
    Parameters:
    -----------
    metric_name : str
        Name of metric
    df : pd.DataFrame, optional
        DataFrame for metric
    top_n : int
        Number of top counties to show
    savepath : str or Path, optional
        Path to save figure
    """
    if df is None:
        df = data[metric_name]
    
    # Sort by mean and get top N
    county_order = df.mean(axis=1).sort_values(ascending=False).index.tolist()
    df_sorted = df.loc[county_order[:top_n]].copy()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create heatmap
    im = ax.imshow(df_sorted.values, aspect='auto', cmap='YlOrRd',
                   vmin=df_sorted.min().min(), vmax=df_sorted.max().max())
    
    # County labels (y-axis)
    ax.set_yticks(range(len(df_sorted.index)))
    ax.set_yticklabels(df_sorted.index, fontsize=10)
    
    # Time labels (x-axis) - show years only
    tick_pos, tick_labs = create_year_tick_labels(time_periods, 'yearly')
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labs, fontsize=11)
    
    # COVID marker
    if covid_idx is not None:
        ax.axvline(x=covid_idx-0.5, color='white', linestyle='--', linewidth=2.5)
    
    # Labels
    ax.set_title(f"{get_metric_display_name(metric_name)} (Top {top_n} Counties)",
                 fontsize=14, weight='bold', pad=15)
    ax.set_xlabel("Year", fontsize=12, weight='bold')
    ax.set_ylabel("County", fontsize=12, weight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(get_metric_display_name(metric_name),
                       rotation=270, labelpad=20, fontsize=11, weight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {savepath.name}")
    plt.show()
    # plt.close(fig)

# COVID Impact Analysis
def plot_covid_impact(metric_name, df=None, savepath=None):
    """
    Two-panel plot comparing pre/during/post COVID periods.
    
    Left: Bar chart of mean values
    Right: Percent change during vs pre
    
    Parameters:
    -----------
    metric_name : str
        Name of metric
    df : pd.DataFrame, optional
        DataFrame for metric
    savepath : str or Path, optional
        Path to save figure
    """
    if df is None:
        df = data[metric_name]
    
    # Classify periods
    pre = [col for col in df.columns if any(y in col for y in ['2018', '2019'])]
    during = [col for col in df.columns if any(y in col for y in ['2020','2021','2022','2023'])]
    post = [col for col in df.columns if '2024' in col]
    
    if not pre or not during or not post:
        print(f"  ⚠ Skipping {metric_name}: insufficient periods")
        return
    
    # Calculate averages
    pre_avg = df[pre].mean(axis=1)
    during_avg = df[during].mean(axis=1)
    post_avg = df[post].mean(axis=1)
    
    # Get top counties by pre-COVID values
    top_counties = pre_avg.sort_values(ascending=False).head(12).index.tolist()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # LEFT PANEL: Bar chart comparison
    x = np.arange(len(top_counties))
    width = 0.26
    
    ax1.bar(x - width, pre_avg[top_counties], width,
            label="Pre (2018-19)", color=COLORS['pre_covid'], alpha=0.9, edgecolor='black', linewidth=0.5)
    ax1.bar(x, during_avg[top_counties], width,
            label="During (2020-23)", color=COLORS['during_covid'], alpha=0.9, edgecolor='black', linewidth=0.5)
    ax1.bar(x + width, post_avg[top_counties], width,
            label="Post (2024)", color=COLORS['post_covid'], alpha=0.9, edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel('County', fontsize=12, weight='bold')
    ax1.set_ylabel(get_metric_display_name(metric_name), fontsize=12, weight='bold')
    ax1.set_title(f'{get_metric_display_name(metric_name)}: COVID-19 Impact Comparison',
                  fontsize=13, weight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(top_counties, rotation=45, ha='right', fontsize=10)
    ax1.legend(frameon=True, loc='best', fontsize=10, shadow=True)
    ax1.grid(axis='y', alpha=0.3, linewidth=0.8)
    
    # RIGHT PANEL: Percent change
    pct_change = ((during_avg - pre_avg) / (pre_avg + 1)) * 100
    top_changes = pct_change.sort_values(ascending=False).head(12)
    colors_change = [COLORS['post_covid'] if v >= 0 else COLORS['covid'] 
                     for v in top_changes.values]
    
    y_pos = np.arange(len(top_changes))
    ax2.barh(y_pos, top_changes.values, color=colors_change, alpha=0.8, 
             edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for j, val in enumerate(top_changes.values):
        ax2.text(val + (2 if val >= 0 else -2), j, f"{val:.1f}%",
                 va='center', ha='left' if val >= 0 else 'right',
                 fontsize=9, weight='bold')
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_changes.index, fontsize=10)
    ax2.set_xlabel('Percent Change (%)', fontsize=12, weight='bold')
    ax2.set_title('Impact on Network Position\n(During COVID vs Pre-COVID)',
                  fontsize=13, weight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1.2)
    ax2.grid(axis='x', alpha=0.3, linewidth=0.8)
    
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {savepath.name}")
    plt.show()
    # plt.close(fig)

# Radar Chart    
def plot_radar_by_season(season_label, counties_to_compare, metrics_to_include, savepath=None):
    """
    Radar chart showing multiple metrics for selected counties across years.
    
    Parameters:
    -----------
    season_label : str
        Season to analyze (e.g., 'Jul', 'Dec')
    counties_to_compare : list
        List of county names
    metrics_to_include : list
        List of metric names
    savepath : str or Path, optional
        Path to save figure
    """
    # Build list of periods for this season
    periods = [p for p in time_periods if f"-{season_label}" in p]
    n_periods = len(periods)
    
    if n_periods == 0:
        print(f"  ⚠ No periods found for season: {season_label}")
        return
    
    fig, axes = plt.subplots(1, n_periods, figsize=(5*n_periods, 5),
                            subplot_kw=dict(projection='polar'))
    if n_periods == 1:
        axes = [axes]
    
    for ax_idx, period in enumerate(periods):
        ax = axes[ax_idx]
        
        # Collect metrics for this period
        snapshot_data = {}
        for metric in metrics_to_include:
            if metric in data and period in data[metric].columns:
                snapshot_data[get_metric_display_name(metric)] = data[metric][period]
        
        if not snapshot_data:
            ax.set_axis_off()
            continue
        
        df_period = pd.DataFrame(snapshot_data)
        df_period = df_period.loc[counties_to_compare]
        
        # Normalize to [0, 1]
        df_norm = (df_period - df_period.min()) / (df_period.max() - df_period.min() + 1e-12)
        
        # Radar geometry
        num_vars = df_norm.shape[1]
        angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        
        # Plot each county
        for county in counties_to_compare:
            if county not in df_norm.index:
                continue
            vals = df_norm.loc[county].values.tolist()
            vals += vals[:1]
            
            ax.plot(angles, vals, marker='o', linewidth=2.5, markersize=6,
                    alpha=0.85, color=county_colors[county], label=county)
            ax.fill(angles, vals, alpha=0.15, color=county_colors[county])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(df_norm.columns, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0.25','0.5','0.75','1.0'], fontsize=8)
        ax.grid(alpha=0.4, linewidth=0.8, linestyle='--')
        
        # Mark COVID periods
        if is_covid_period(period):
            ax.set_title(period + "\n(COVID)", fontsize=11, fontweight='bold', color=COLORS['covid'])
        else:
            ax.set_title(period, fontsize=11, fontweight='bold')
    
    # Common legend
    handles = [mpl.lines.Line2D([0],[0], color=county_colors[c], marker='o',
                                linewidth=2.5, markersize=6, alpha=0.85)
               for c in counties_to_compare]
    
    fig.legend(handles, counties_to_compare, loc='lower center', ncol=min(len(counties_to_compare), 6),
               frameon=True, fontsize=11, title=f"Counties ({season_label})", title_fontsize=12, shadow=True)
    
    fig.suptitle(f"Multi-Metric Radar Comparison: {season_label} Snapshots (2018-2024)\\n" +
                 "Metrics normalized within each year",
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {savepath.name}")
    plt.show()
    # plt.close(fig)


# Correlation Evolution
def plot_correlation_evolution(metrics_list, savepath=None):
    """
    Show how metric correlations evolve over time (focusing on July each year).
    
    Parameters:
    -----------
    metrics_list : list
        List of metric names to include
    savepath : str or Path, optional
        Path to save figure
    """
    # Focus on July snapshots
    key_periods = [p for p in time_periods if '-Jul' in p]
    
    n_periods = len(key_periods)
    if n_periods == 0:
        return
    
    ncols = 3
    nrows = (n_periods + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5*nrows))
    axes = axes.flatten() if n_periods > 1 else [axes]
    
    for idx, period in enumerate(key_periods):
        ax = axes[idx]
        
        # Collect metrics for this period
        corr_data = {}
        for metric in metrics_list:
            if metric in data and period in data[metric].columns:
                corr_data[get_metric_display_name(metric)] = data[metric][period].values
        
        if not corr_data:
            ax.set_axis_off()
            continue
        
        corr_df = pd.DataFrame(corr_data, index=counties)
        corr_mat = corr_df.corr()
        
        # Plot heatmap
        im = ax.imshow(corr_mat.values, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        
        n_metrics = len(corr_mat)
        ax.set_xticks(range(n_metrics))
        ax.set_yticks(range(n_metrics))
        ax.set_xticklabels(corr_mat.columns, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(corr_mat.index, fontsize=9)
        
        # Add correlation values
        for i in range(n_metrics):
            for j in range(i+1, n_metrics):
                val = corr_mat.iloc[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="white" if abs(val) > 0.5 else "black", fontsize=9, weight='bold')
        
        # Mark COVID periods
        if is_covid_period(period):
            ax.set_title(period + " (COVID)", fontsize=11, weight='bold', color=COLORS['covid'])
            for spine in ax.spines.values():
                spine.set_edgecolor(COLORS['covid'])
                spine.set_linewidth(3)
        else:
            ax.set_title(period, fontsize=11, weight='bold')
    
    # Hide extra subplots
    for j in range(n_periods, len(axes)):
        axes[j].set_axis_off()
    
    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=-1, vmax=1), cmap='coolwarm'),
        cax=cbar_ax
    )
    cbar.set_label("Correlation Coefficient", rotation=270, labelpad=25, fontsize=12, weight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    fig.suptitle("Metric Correlation Evolution Over Time (July Snapshots)",
                 fontsize=15, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {savepath.name}")
    plt.show()
    # plt.close(fig)

# Normalized Comparison Single County
def plot_normalized_comparison(metrics_list, county_name, savepath=None):
    """
    Compare multiple metrics for a single county (all normalized to [0,1]).
    
    Parameters:
    -----------
    metrics_list : list
        List of metric names
    county_name : str
        Name of county
    savepath : str or Path, optional
        Path to save figure
    """
    if county_name not in counties:
        print(f"  ⚠ County '{county_name}' not found")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    metric_colors_local = plt.get_cmap("Set2").colors
    
    for idx, metric in enumerate(metrics_list):
        if metric not in data:
            continue
        
        vals = data[metric].loc[county_name].values
        norm_vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-10)
        
        ax.plot(time_numeric, norm_vals, marker='o', linewidth=2.5, markersize=5,
                label=get_metric_display_name(metric),
                color=metric_colors_local[idx % len(metric_colors_local)], alpha=0.9)
    
    # COVID marker and shading
    if covid_idx is not None:
        ax.axvline(x=covid_idx, color=COLORS['covid'], linestyle='--', linewidth=2, alpha=0.7)
        ax.axvspan(covid_idx-0.5, covid_idx+8, alpha=0.1, color=COLORS['covid'], zorder=0)
    
    # Year boundaries
    year_bounds = {}
    for idx, period in enumerate(time_periods):
        year = period.split('-')[0]
        if year not in year_bounds:
            year_bounds[year] = idx
    
    for year, idx in list(year_bounds.items())[1:]:
        ax.axvline(x=idx-0.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.3, zorder=0)
    
    # Labels
    ax.set_xlabel('Time Period', fontsize=12, weight='bold')
    ax.set_ylabel('Normalized Value (0-1)', fontsize=12, weight='bold')
    ax.set_title(f'Network Metrics Evolution: {county_name} County',
                 fontsize=14, weight='bold', pad=15)
    ax.set_ylim(-0.05, 1.05)
    
    # Clean x-axis
    tick_pos, tick_labs = create_year_tick_labels(time_periods, 'yearly')
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labs, fontsize=11)
    
    ax.legend(loc='best', frameon=True, ncol=2, fontsize=10,
              title="Metric", title_fontsize=11, shadow=True)
    ax.grid(axis='y', alpha=0.3, linewidth=0.8)
    
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {savepath.name}")
    plt.show()
    # plt.close(fig)


# Batch Plotting Functions
def generate_all_timeseries(metrics_list=None, top_n=10):
    """Generate time series plots for all specified metrics"""
    if metrics_list is None:
        from config import KEY_METRICS
        metrics_list = KEY_METRICS
    
    for metric in metrics_list:
        if metric not in data:
            continue
    
        
        # Top N counties
        plot_timeseries_top(
            metric, top_n=top_n,
            savepath=OUTPUT_DIR / f"TS_{metric}_Top{top_n}.png"
        )
        # All counties
        plot_timeseries_all(
            metric,
            savepath=OUTPUT_DIR / f"TS_{metric}_All.png"
        )

def generate_all_heatmaps(metrics_list=None, top_n=15):
    """Generate heatmap plots for all specified metrics"""
    if metrics_list is None:
        from config import KEY_METRICS
        metrics_list = KEY_METRICS

    for metric in metrics_list:
        if metric not in data:
            continue
        plot_heatmap(
            metric, top_n=top_n,
            savepath=OUTPUT_DIR / f"Heatmap_{metric}_Top{top_n}.png"
        )



def generate_all_covid_plots(metrics_list=None):
    """Generate COVID impact plots for all specified metrics"""
    if metrics_list is None:
        from config import KEY_METRICS
        metrics_list = KEY_METRICS

    for metric in metrics_list:
        if metric not in data:
            continue
        
        plot_covid_impact(
            metric,
            savepath=OUTPUT_DIR / f"COVID_Impact_{metric}.png"
        )
    

# Main execution
# if __name__ == "__main__":
#     from config import setup_plot_style, KEY_METRICS, CORRELATION_METRICS
    
#     # Setup
#     setup_plot_style()
#     load_data()
    
#     # Generate all standard plots
#     generate_all_timeseries(KEY_METRICS, top_n=10)
#     generate_all_heatmaps(KEY_METRICS, top_n=15)
#     generate_all_covid_plots(KEY_METRICS)
    
#     # Correlation evolution
#     plot_correlation_evolution(
#         CORRELATION_METRICS,
#         savepath=OUTPUT_DIR / "Correlation_Evolution_July.png"
#     )
    
#     # Radar chart
#     plot_radar_by_season(
#         season_label='Jul',
#         counties_to_compare=['Laramie', 'Teton', 'Park', 'Natrona', 'Fremont'],
#         metrics_to_include=['InDegree', 'PageRank', 'Betweenness', 'Eigenvector'],
#         savepath=OUTPUT_DIR / "Radar_July_TopCounties.png"
#     )
    
#     # Normalized ormalized comparison
#     plot_normalized_comparison(
#         metrics_list=['InDegree', 'PageRank', 'Betweenness', 'Closeness'],
#         county_name='Laramie',
#         savepath=OUTPUT_DIR / "Normalized_Laramie.png"
#     )
  