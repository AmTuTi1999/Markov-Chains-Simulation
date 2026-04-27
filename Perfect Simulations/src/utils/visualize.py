"""
Visualization module for perfect simulation experiments.

Contains comprehensive plotting functions for:
- Bound validation and comparison
- Tightness analysis
- Tail probabilities
- Confidence intervals
- Algorithm comparisons
"""

import matplotlib.pyplot as plt
import numpy as np
import os


# ============================================================================
# BASIC PLOTTING UTILITIES
# ============================================================================

def plot_and_save_figure(x, y, z=None, xlabel="", ylabel="", title="", 
                         filename=None, figsize=(7, 5), **kwargs):
    """
    Generic plotting utility for single or dual-curve comparison.
    
    Parameters
    ----------
    x : dict or array-like
        X-axis values. If dict, keys are used (sorted).
    y : array-like
        Y-axis values (first curve).
    z : array-like, optional
        Y-axis values (second curve).
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    title : str
        Plot title.
    filename : str, optional
        If provided, save figure to this file.
    figsize : tuple
        Figure size (width, height).
    **kwargs : dict
        Additional keyword arguments:
        - label_1: Label for first curve
        - label_2: Label for second curve
        - marker_1: Marker style for first curve (default 'o')
        - marker_2: Marker style for second curve (default 's')
        - linestyle_1: Line style for first curve (default '-')
        - linestyle_2: Line style for second curve (default '--')
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # Normalize x input
    if isinstance(x, dict):
        x_vals = np.array(sorted(x.keys()), dtype=float)
    else:
        x_vals = np.array(x, dtype=float)
    
    y_vals = np.array(y, dtype=float)
    
    if len(x_vals) != len(y_vals):
        raise ValueError(f"x and y must have same length: {len(x_vals)} vs {len(y_vals)}")
    
    if z is not None:
        z_vals = np.array(z, dtype=float)
        if len(z_vals) != len(x_vals):
            raise ValueError(f"x and z must have same length: {len(x_vals)} vs {len(z_vals)}")
    else:
        z_vals = None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot first curve
    marker_1 = kwargs.get('marker_1', 'o')
    linestyle_1 = kwargs.get('linestyle_1', '-')
    label_1 = kwargs.get('label_1', None)
    
    ax.plot(x_vals, y_vals, marker=marker_1, linestyle=linestyle_1,
            linewidth=2, markersize=6, label=label_1)
    
    # Plot second curve if provided
    if z_vals is not None:
        marker_2 = kwargs.get('marker_2', 's')
        linestyle_2 = kwargs.get('linestyle_2', '--')
        label_2 = kwargs.get('label_2', None)
        
        ax.plot(x_vals, z_vals, marker=marker_2, linestyle=linestyle_2,
                linewidth=2, markersize=6, label=label_2)
    
    # Styling
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)
    
    if label_1 or label_2:
        ax.legend(frameon=True)
    
    fig.tight_layout()
    
    # Save if filename provided
    if filename:
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
    
    return fig


def plot_and_save_figure_gallo(x, y, z, xlabel, ylabel, title, **kwargs):
    """Legacy function for compatibility. Use plot_and_save_figure instead."""
    return plot_and_save_figure(x, y, z, xlabel, ylabel, title, **kwargs)


# ============================================================================
# BOUND VALIDATION PLOTS
# ============================================================================

def plot_bound_validation(theoretical, mc, title, filename=None, figsize=(5.2, 5.2)):
    """
    Scatter plot comparing theoretical bounds vs MC empirical estimates.
    Includes perfect agreement line y=x.
    
    Parameters
    ----------
    theoretical : array-like
        Theoretical bound values.
    mc : array-like
        Monte Carlo empirical estimates.
    title : str
        Plot title.
    filename : str, optional
        If provided, save figure to this file.
    figsize : tuple
        Figure size.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    theoretical = np.array(theoretical)
    mc = np.array(mc)
    
    # Scatter plot
    ax.scatter(theoretical, mc, s=50, alpha=0.7)
    
    # Perfect agreement line
    minv = min(theoretical.min(), mc.min())
    maxv = max(theoretical.max(), mc.max())
    ax.plot([minv, maxv], [minv, maxv], 'k--', linewidth=2, label='Perfect agreement')
    
    # Labels and styling
    ax.set_xlabel('Theoretical bound', fontsize=12)
    ax.set_ylabel(r'MC $\hat{\mathbb{E}}[L_n]$', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    if filename:
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
    
    return fig


# ============================================================================
# TIGHTNESS PLOTS
# ============================================================================

def plot_tightness_comparison(x1, y1, x2, y2, label1='Exponential decay', 
                              label2='Polynomial decay', filename=None):
    """
    Compare tightness across two decay types.
    
    Parameters
    ----------
    x1, y1 : array-like
        X and Y values for first curve.
    x2, y2 : array-like
        X and Y values for second curve.
    label1, label2 : str
        Curve labels.
    filename : str, optional
        Save path.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    ax.plot(x1, y1, 'o-', linewidth=2, markersize=6, label=label1)
    ax.plot(x2, y2, 's--', linewidth=2, markersize=6, label=label2)
    
    # Ideal bound line
    ax.axhline(100, linestyle=':', linewidth=2, label='Perfect bound (100%)')
    
    ax.set_xlabel('Load parameter (ρ) / Tail parameter (p)', fontsize=12)
    ax.set_ylabel('Tightness (%)', fontsize=12)
    ax.set_title('Tightness of Analytical Bound', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    
    fig.tight_layout()
    
    if filename:
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
    
    return fig


def plot_single_tightness(x, y, xlabel, title, filename=None, figsize=(6, 4.2),
                          marker='o', linestyle='-'):
    """
    Plot tightness for a single decay type.
    
    Parameters
    ----------
    x, y : array-like
        X and Y values.
    xlabel : str
        X-axis label.
    title : str
        Plot title.
    filename : str, optional
        Save path.
    figsize : tuple
        Figure size.
    marker : str
        Marker style.
    linestyle : str
        Line style.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(x, y, marker=marker, linestyle=linestyle, linewidth=2, markersize=6)
    ax.axhline(100, linestyle=':', linewidth=2, label='Perfect bound')
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Tightness (%)', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(90, 112)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    
    fig.tight_layout()
    
    if filename:
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
    
    return fig


# ============================================================================
# TAIL PROBABILITY PLOTS
# ============================================================================

def plot_tail_probability(x1, y1, x2, y2, label1='Exponential decay',
                         label2='Polynomial decay', filename=None):
    """
    Plot tail probabilities on log scale for multiple decay types.
    
    Parameters
    ----------
    x1, y1 : array-like
        X and Y values for first curve.
    x2, y2 : array-like
        X and Y values for second curve.
    label1, label2 : str
        Curve labels.
    filename : str, optional
        Save path.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    ax.semilogy(x1, y1, 'o-', linewidth=2, markersize=6, label=label1)
    ax.semilogy(x2, y2, 's--', linewidth=2, markersize=6, label=label2)
    
    ax.set_xlabel(r'Parameter ($\rho$ or $p$)', fontsize=12)
    ax.set_ylabel(r'Tail probability $P(L_n > S)$ (log scale)', fontsize=12)
    ax.set_title('Truncation Tail Probability', fontsize=13)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(frameon=True)
    
    fig.tight_layout()
    
    if filename:
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
    
    return fig


# ============================================================================
# DUAL-AXIS PLOTS
# ============================================================================

def plot_tightness_and_tail(x, tight, tail, xlabel, title, filename=None, figsize=(6.5, 4.5)):
    """
    Plot tightness and tail probability on dual y-axes.
    
    Parameters
    ----------
    x : array-like
        X-axis values.
    tight : array-like
        Tightness values (left axis).
    tail : array-like
        Tail probabilities (right axis, log scale).
    xlabel : str
        X-axis label.
    title : str
        Plot title.
    filename : str, optional
        Save path.
    figsize : tuple
        Figure size.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Left axis: tightness
    ax1.plot(x, tight, 'o-', linewidth=2, markersize=6, label='Tightness (%)')
    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel('Tightness (%)', fontsize=12)
    ax1.set_ylim(90, 115)
    ax1.grid(True, alpha=0.3)
    
    # Right axis: tail probability
    ax2 = ax1.twinx()
    ax2.plot(x, tail, 's--', linewidth=2, color='red', label=r'$P(L_n>S)$')
    ax2.set_yscale('log')
    ax2.set_ylabel(r'Tail probability $P(L_n>S)$ (log scale)', fontsize=12)
    
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    
    if filename:
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
    
    return fig


# ============================================================================
# CONFIDENCE INTERVAL PLOTS
# ============================================================================

def plot_empirical_with_ci(x, empirical, ci_low, ci_high, xlabel, ylabel, 
                          title, filename=None, figsize=(7, 5)):
    """
    Plot empirical estimates with confidence interval bands.
    
    Parameters
    ----------
    x : array-like
        X-axis values (e.g., alpha values).
    empirical : array-like
        Empirical point estimates.
    ci_low : array-like
        Lower confidence interval bounds.
    ci_high : array-like
        Upper confidence interval bounds.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    title : str
        Plot title.
    filename : str, optional
        Save path.
    figsize : tuple
        Figure size.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(x, empirical, 'o-', linewidth=2, markersize=6, label='Empirical')
    ax.fill_between(x, ci_low, ci_high, alpha=0.3, label='95% CI')
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    
    fig.tight_layout()
    
    if filename:
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
    
    return fig


# ============================================================================
# GLOBAL BOUND ANALYSIS
# ============================================================================

def plot_global_tightness(x, tight, xlabel, title, filename=None, figsize=(6, 4.2),
                         marker='o', linestyle='-'):
    """
    Plot global (non-truncated) tightness.
    
    Parameters
    ----------
    x : array-like
        X-axis values (load or tail parameter).
    tight : array-like
        Tightness values (%).
    xlabel : str
        X-axis label.
    title : str
        Plot title.
    filename : str, optional
        Save path.
    figsize : tuple
        Figure size.
    marker : str
        Marker style.
    linestyle : str
        Line style.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(x, tight, marker=marker, linestyle=linestyle, linewidth=2, markersize=6)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Tightness (%)', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    if filename:
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
    
    return fig


# ============================================================================
# LOG-SCALE ANALYSIS
# ============================================================================

def plot_log_scale(x, y, xlabel, ylabel, title, filename=None, figsize=(7, 5),
                  yscale='log', marker='o', linestyle='-'):
    """
    Plot data on logarithmic scale.
    
    Parameters
    ----------
    x, y : array-like
        X and Y data.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    title : str
        Plot title.
    filename : str, optional
        Save path.
    figsize : tuple
        Figure size.
    yscale : str
        Y-axis scale ('log' or 'loglog').
    marker : str
        Marker style.
    linestyle : str
        Line style.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(x, y, marker=marker, linestyle=linestyle, linewidth=2, markersize=6)
    ax.set_yscale(yscale)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, which='both', alpha=0.3)
    
    fig.tight_layout()
    
    if filename:
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
    
    return fig


# ============================================================================
# TAIL REMAINDER ANALYSIS
# ============================================================================

def plot_tail_remainder_normalized(x1, y1, x2, y2, label1='Exponential decay',
                                  label2='Polynomial decay', filename=None):
    """
    Plot tail remainder with normalized x-axes for comparison.
    
    Parameters
    ----------
    x1, y1 : array-like
        First curve data.
    x2, y2 : array-like
        Second curve data.
    label1, label2 : str
        Curve labels.
    filename : str, optional
        Save path.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # Normalize x-axes to [0, 1]
    x1 = np.array(x1)
    x2 = np.array(x2)
    x1_norm = (x1 - x1.min()) / (x1.max() - x1.min())
    x2_norm = (x2 - x2.min()) / (x2.max() - x2.min())
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    ax.semilogy(x1_norm, y1, 'o-', linewidth=2, markersize=6, label=label1)
    ax.semilogy(x2_norm, y2, 's--', linewidth=2, markersize=6, label=label2)
    
    ax.set_ylabel(r'$\mathbb{E}[(L_n-S)_+]$', fontsize=12)
    ax.set_title('Tail remainder comparison', fontsize=13)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(frameon=True)
    
    fig.tight_layout()
    
    if filename:
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
    
    return fig


# ============================================================================
# BIAS ANALYSIS
# ============================================================================

def plot_bias(x, bias, xlabel, title, filename=None, figsize=(6, 4.2)):
    """
    Plot bias values across parameter range.
    
    Parameters
    ----------
    x : array-like
        Parameter values (e.g., alpha).
    bias : array-like
        Bias values.
    xlabel : str
        X-axis label.
    title : str
        Plot title.
    filename : str, optional
        Save path.
    figsize : tuple
        Figure size.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(x, bias, 'o-', linewidth=2, markersize=6)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Bias', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    if filename:
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
    
    return fig


# ============================================================================
# BATCH PLOTTING UTILITIES
# ============================================================================

def save_all_plots(plots_dict, output_dir):
    """
    Save multiple figures to output directory.
    
    Parameters
    ----------
    plots_dict : dict
        Dictionary of {filename: fig} pairs.
    output_dir : str
        Output directory path.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for filename, fig in plots_dict.items():
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")


def close_all_plots(plots_dict):
    """Close all figures in dictionary."""
    for fig in plots_dict.values():
        plt.close(fig)
