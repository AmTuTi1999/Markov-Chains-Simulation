# Visualization Integration Guide

This document explains how the visualization module has been integrated into the pipeline files to automatically generate plots during simulation runs.

## Overview

All plotting functionality from `notebooks/test_plots.ipynb` has been moved to `src/utils/visualize.py`. The pipeline files now import and use these functions to generate plots directly as simulations complete.

## New Visualization Module Structure

### Location
```
Perfect Simulations/
└── src/utils/
    └── visualize.py
```

### Function Categories

The visualization module contains the following categories of plotting functions:

#### 1. **Basic Plotting Utilities**
- `plot_and_save_figure()` - Generic dual-curve comparison plot
- `plot_and_save_figure_gallo()` - Legacy wrapper for compatibility

#### 2. **Bound Validation**
- `plot_bound_validation()` - Scatter plot comparing theoretical bounds vs MC empirical estimates

#### 3. **Tightness Analysis**
- `plot_tightness_comparison()` - Compare tightness across decay types
- `plot_single_tightness()` - Single decay type tightness
- `plot_global_tightness()` - Global (non-truncated) tightness

#### 4. **Tail Probability**
- `plot_tail_probability()` - Log-scale tail probability comparison
- `plot_tightness_and_tail()` - Dual-axis plot of tightness and tail

#### 5. **Confidence Intervals**
- `plot_empirical_with_ci()` - Empirical estimates with 95% CI bands

#### 6. **Log-Scale Analysis**
- `plot_log_scale()` - Flexible log-scale plotting

#### 7. **Tail Remainder**
- `plot_tail_remainder_normalized()` - Normalized x-axis comparison

#### 8. **Bias Analysis**
- `plot_bias()` - Bias across parameter ranges

#### 9. **Batch Utilities**
- `save_all_plots()` - Save multiple figures at once
- `close_all_plots()` - Close multiple figures


## Pipeline Integration

### Updated Files

All three pipeline files have been updated to:
1. Import visualization functions from the module
2. Call these functions within their plot generation routines

#### Files Modified:
- `Perfect Simulations/src/pipelines/context_tree_kernel_pipeline.py`
- `Perfect Simulations/src/pipelines/continuous_kernel_pipelines.py`
- `Perfect Simulations/src/pipelines/locally_continuous_kernel_pipeline.py`

### Import Statement

Each pipeline now includes:

```python
from src.utils.visualize import (
    plot_and_save_figure,
    plot_bound_validation,
    plot_tightness_and_tail,
    plot_tail_probability,
    plot_bias,
    plot_log_scale,
)
```

## Usage Examples

### Example 1: Basic Tightness Comparison

```python
from src.utils.visualize import plot_tightness_comparison

# Plot tightness for exponential vs polynomial decay
rho = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
tight_exp = np.array([94.69, 97.33, 100, 100, 100])
p = np.array([3, 4, 5, 6, 7])
tight_poly = np.array([98.34, 99.01, 101.47, 105.29, 98.18])

fig = plot_tightness_comparison(
    rho, tight_exp, p, tight_poly,
    filename='output_dir/tightness_comparison.pdf'
)
```

### Example 2: Bound Validation

```python
from src.utils.visualize import plot_bound_validation

theoretical = np.array([0.1166, 0.2845, 0.5544, 0.9894, 1.7932])
mc = np.array([0.1104, 0.2769, 0.5544, 0.9928, 1.7932])

fig = plot_bound_validation(
    theoretical, mc,
    title='Truncation Induced Bound Validation',
    filename='output_dir/validation.pdf'
)
```

### Example 3: Empirical with Confidence Intervals

```python
from src.utils.visualize import plot_empirical_with_ci

alpha = np.array([0.005, 0.008, 0.011, 0.013, 0.016])
empirical = np.array([1.021573, 1.021573, 1.021573, 1.021573, 1.021573])
ci_low = np.array([1.020690, 1.020690, 1.020690, 1.020690, 1.020690])
ci_high = np.array([1.022456, 1.022456, 1.022456, 1.022456, 1.022456])

fig = plot_empirical_with_ci(
    alpha, empirical, ci_low, ci_high,
    xlabel=r'$\alpha$',
    ylabel=r'$\mathbb{E}[L]$',
    title='Empirical Estimate with 95% CI',
    filename='output_dir/ci_plot.pdf'
)
```

### Example 4: Dual-Axis Plot

```python
from src.utils.visualize import plot_tightness_and_tail

rho = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
tight = np.array([94.69, 97.33, 100, 100, 100, 98.70, 100, 99.32, 99.55])
tail = np.array([1e0, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0, 7.34e-228])

fig = plot_tightness_and_tail(
    rho, tight, tail,
    xlabel=r'Load $\rho$',
    title='Truncated Case — Exponential Decay',
    filename='output_dir/dual_axis.pdf'
)
```

## Automatic Plot Generation in Pipelines

The pipelines now automatically generate plots during execution. For example, in `context_tree_kernel_pipeline.py`:

```python
# Results storage
results = {
    "truncated": {...},
    "non_truncated": {...}
}

# ... run simulations and fill results ...

# Plots are generated automatically
if run_truncated:
    _generate_truncated_plots(results["truncated"], results_dir, ...)

if run_non_truncated:
    _generate_non_truncated_plots(results["non_truncated"], results_dir, ...)
```

## Output Structure

Plots are saved to the `results/` directory organized by algorithm:

```
results/
├── gallo/
│   ├── Truncated_Empirical_vs_Analytical_*.pdf
│   ├── Truncated_Tightness_*.pdf
│   ├── NonTruncated_Tightness_*.pdf
│   └── ... (other plots)
├── garcia/
│   └── ... (Garcia-specific plots)
└── cff/
    └── ... (CFF-specific plots)
```

## Notebook to Pipeline Migration

The migration from notebook-based plotting (`notebooks/test_plots.ipynb`) to pipeline-integrated plotting involved:

1. **Extraction**: All plotting code from the notebook was analyzed and categorized
2. **Generalization**: Plotting functions were made flexible and reusable
3. **Organization**: Functions were grouped by plot type for easy navigation
4. **Documentation**: Comprehensive docstrings added to each function
5. **Integration**: Pipeline files updated to call visualization functions
6. **Compatibility**: Legacy function names preserved for backward compatibility

## Benefits

1. **Automatic Plots**: Plots generate automatically without manual notebook execution
2. **Consistency**: Same plotting code used across all simulations
3. **Reusability**: Functions can be used independently or in batch
4. **Maintainability**: Single source of truth for all visualization logic
5. **Scalability**: Easy to add new plot types without modifying pipelines

## Adding New Plots

To add a new plot type:

1. Add the function to `src/utils/visualize.py` following the existing pattern
2. Import the function in the relevant pipeline file
3. Call the function in the appropriate plot generation routine
4. Save the output to the results directory

Example template:

```python
def plot_my_custom_analysis(x, y, title, filename=None, figsize=(7, 5)):
    """
    Brief description of the plot.
    
    Parameters
    ----------
    x : array-like
        X-axis data
    y : array-like
        Y-axis data
    title : str
        Plot title
    filename : str, optional
        Save path
    figsize : tuple
        Figure size
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Your plotting code here
    ax.plot(x, y, 'o-', linewidth=2, markersize=6)
    
    # Styling
    ax.set_xlabel('X Label', fontsize=12)
    ax.set_ylabel('Y Label', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    if filename:
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
    
    return fig
```

## Running Simulations with Automatic Plot Generation

To run simulations that automatically generate all plots:

```bash
# From Perfect Simulations directory
python simulate_gallo.py          # Gallo with automatic plots
python simulate_garcia.py         # Garcia with automatic plots
python simulate_cff.py            # CFF with automatic plots
python run_unified_experiments.py # All algorithms with plots
```

All plots will be saved to `results/{algorithm}/` automatically.

## Questions and Support

For questions about specific plot functions, refer to their docstrings:

```python
from src.utils.visualize import plot_tightness_comparison
help(plot_tightness_comparison)
```

For integration questions, see the pipeline files for usage examples.
