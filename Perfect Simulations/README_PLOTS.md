# Quick Reference: Plot Functionalities Migration

## What Happened

✅ All plotting code from `notebooks/test_plots.ipynb` has been moved to the pipeline system.
✅ Plots now generate automatically when you run simulations.
✅ No more manual notebook execution needed!

## Files Created/Modified

### New Files (3)
1. **Enhanced `src/utils/visualize.py`**
   - 950+ lines of reusable plotting functions
   - 15+ functions organized by category
   - Full docstrings and examples

2. **`VISUALIZATION_GUIDE.md`** (Complete API Reference)
   - All functions documented
   - Usage examples for each plot type
   - Integration guidelines

3. **`example_visualization.py`** (Runnable Examples)
   - Tests all visualization functions
   - Generates 12 example plots
   - Great for learning the API

### Modified Files (3)
1. **`src/pipelines/context_tree_kernel_pipeline.py`**
   - Now imports from visualization module
   - Removed duplicate code

2. **`src/pipelines/continuous_kernel_pipelines.py`**
   - Enhanced imports

3. **`src/pipelines/locally_continuous_kernel_pipeline.py`**
   - Enhanced imports

## How to Use

### Option 1: Automatic (Recommended)
Just run the simulations - plots generate automatically!

```bash
cd Perfect\ Simulations
python3 simulate_gallo.py       # ← Plots generated automatically
python3 simulate_garcia.py      # ← Plots generated automatically
python3 simulate_cff.py         # ← Plots generated automatically
```

### Option 2: Run Examples
Test all visualizations with sample data:

```bash
cd Perfect\ Simulations
python3 example_visualization.py
```

This generates 12 example plots in `results/example_*.pdf`

### Option 3: Use in Your Code
```python
from src.utils.visualize import plot_tightness_comparison
import numpy as np

# Your data
rho = np.array([0.1, 0.2, 0.3])
tight = np.array([94.69, 97.33, 102.36])

# Create plot (saved to file)
fig = plot_tightness_comparison(
    rho, tight, ...,
    filename='my_plot.pdf'
)
```

## Available Functions

### Basic Plotting
- `plot_and_save_figure()` - Generic dual-curve plot
- `save_all_plots()` - Save multiple figures
- `close_all_plots()` - Close all figures

### Bound Validation
- `plot_bound_validation()` - Theoretical vs empirical scatter

### Tightness Analysis
- `plot_tightness_comparison()` - Compare decay types
- `plot_single_tightness()` - Single decay type
- `plot_global_tightness()` - Non-truncated bounds

### Tail Probabilities
- `plot_tail_probability()` - Log-scale comparison
- `plot_tightness_and_tail()` - Dual-axis plot

### Other Analysis
- `plot_empirical_with_ci()` - With confidence intervals
- `plot_log_scale()` - Flexible log-scale
- `plot_tail_remainder_normalized()` - Normalized comparison
- `plot_bias()` - Bias analysis

## Output Locations

Plots are saved to:
```
results/
├── gallo/           (Gallo algorithm plots)
├── garcia/          (Garcia algorithm plots)
├── cff/             (CFF algorithm plots)
└── example_*.pdf    (Example plots from example_visualization.py)
```

## Documentation

1. **VISUALIZATION_GUIDE.md** ← Read this for complete API
2. **MIGRATION_SUMMARY.md** ← Read this for migration details
3. **example_visualization.py** ← Read this for code examples
4. **src/utils/visualize.py** ← Full source code with docstrings

## Key Changes for Users

### Before (Old Way - Still Works)
```python
# Had to run notebook manually
# Plots were in notebook output
# Hard to reuse code
```

### Now (New Way)
```python
# Simulations auto-generate plots
# Plots saved to results/ directory
# Code is modular and reusable
```

## Verification

All functions have been tested:
```bash
python3 example_visualization.py  # ✓ All 12 plots generated successfully
```

## Questions?

Check these files in order:
1. `VISUALIZATION_GUIDE.md` - For function signatures
2. `example_visualization.py` - For code examples
3. `src/utils/visualize.py` - For implementation details

---

**Status**: ✅ Complete and tested
**Ready for production**: ✅ Yes
**Breaking changes**: ❌ None (backward compatible)
