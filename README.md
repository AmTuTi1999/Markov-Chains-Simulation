# Markov Chains Perfect Simulation

A comprehensive implementation of perfect simulation algorithms for Markov chains, including implementations of Gallo (2009), Garcia (2011), and CFF (Continuous-time Fast Forwarding) methods.

## Overview

This repository contains implementations of advanced perfect simulation techniques for Markov chains with applications to locally continuous kernels and context trees. Perfect simulation provides unbiased samples from the stationary distribution without requiring burn-in periods or truncation.

## Project Structure

```
Markov-Chains-Simulation/
├── Perfect Simulations/          # Main implementation directory
│   ├── src/
│   │   ├── simulators/          # Core simulator implementations
│   │   │   ├── perfect_simulator.py    # Base class for all simulators
│   │   │   ├── gallo.py               # Gallo (2009) algorithm
│   │   │   ├── garcia.py              # Garcia (2011) algorithm
│   │   │   ├── cff.py                 # CFF method
│   │   │   ├── utils.py               # Helper functions
│   │   │   └── README.md              # Detailed algorithm documentation
│   │   ├── pipelines/           # Data processing pipelines
│   │   │   ├── context_tree_kernel_pipeline.py
│   │   │   ├── continuous_kernel_pipelines.py
│   │   │   └── locally_continuous_kernel_pipeline.py
│   │   └── utils/               # Utility modules
│   │       ├── context_tree.py
│   │       ├── decay.py
│   │       └── visualize.py
│   ├── simulate_gallo.py        # Gallo simulation runner
│   ├── simulate_garcia.py       # Garcia simulation runner
│   ├── simulate_cff.py          # CFF simulation runner
│   ├── compare_gallo_garcia.py  # Comparison script
│   └── run_unified_experiments.py  # Full experiment suite
├── results/                     # Numerical results and benchmarks
│   ├── gallo/                   # Gallo algorithm comparisons
│   ├── garcia/                  # Garcia algorithm results
│   └── cff/                     # CFF method results
├── notebooks/                   # Jupyter notebooks for analysis
└── README.md                    # This file
```

## Features

- **Multiple Perfect Simulation Algorithms**
  - Gallo (2009): Distance-based coalescence detection
  - Garcia (2011): Block-based independence structure for locally continuous kernels
  - CFF: Continuous-time fast forwarding methods

- **Flexible Implementation**
  - Works with arbitrary finite Markov chains
  - Supports context trees and locally continuous kernels
  - Extensible base classes for new algorithms

- **Comprehensive Benchmarking**
  - Compare algorithm performance across different parameters
  - Numerical results and statistics
  - Visualization tools for convergence analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AmTuTi1999/Markov-Chains-Simulation.git
cd Markov-Chains-Simulation
```

2. Install required dependencies:
```bash
pip install numpy scipy matplotlib tqdm
```

3. (Optional) For Jupyter notebook analysis:
```bash
pip install jupyter ipython
```

## Quick Start

### Basic Perfect Sampling (Markov Chain)

```python
import numpy as np
from Perfect_Simulations.perfect_simulation_for_one_order_mc import PerfectSampler

# Create a simple 2-state Markov chain
transition_matrix = np.array([
    [0.7, 0.3],
    [0.4, 0.6]
])

# Initialize sampler
sampler = PerfectSampler(transition_matrix, seed=42)

# Get a perfect sample from the stationary distribution
sample = sampler.perfect_sample(n=0)
print(f"Perfect sample: {sample}")
```

### Running Simulation Experiments

```bash
# Run Gallo simulation
python Perfect\ Simulations/simulate_gallo.py

# Run Garcia simulation
python Perfect\ Simulations/simulate_garcia.py

# Run CFF simulation
python Perfect\ Simulations/simulate_cff.py

# Compare multiple algorithms
python Perfect\ Simulations/compare_gallo_garcia.py

# Run full experiment suite
python Perfect\ Simulations/run_unified_experiments.py
```

### Example: Compare Algorithms

```python
from Perfect_Simulations.compare_gallo_garcia import compare_garcia_vs_gallo

# Run comparison with custom parameters
results = compare_garcia_vs_gallo(
    num_validation_samples=10000,
    alpha_values=[0.001, 0.01, 0.05],
    max_time=1000
)

# Results include timing, sample quality, and convergence metrics
print(f"Gallo execution time: {results['gallo_time']:.3f}s")
print(f"Garcia execution time: {results['garcia_time']:.3f}s")
```

## Key Algorithms

### Gallo (2009)
Distance-based coalescence detection using a reference string. Detects when all paths have "coalesced" by being within a distance threshold from a reference string.

**Advantages:**
- Simple to implement
- Works for general Markov chains
- Can be truncated for faster computation

**Trade-offs:**
- Bias if truncated

### Garcia (2011)
Block-based algorithm exploiting independence structure of locally continuous kernels.

**Advantages:**
- True perfect simulation (unbiased)
- Block independence enables renewal analysis
- No truncation needed

**Best for:**
- Locally continuous kernels
- Situations where exact samples are critical

### CFF (Continuous Fast Forwarding)
Continuous-time perfect simulation method using exponential clocks and fast forwarding.

**Advantages:**
- Efficient for continuous-time chains
- Natural formulation for intensity-based models

## Results and Benchmarks

Numerical results are stored in the `results/` directory, organized by algorithm:

- `results/gallo/`: Gallo algorithm benchmarks for different decay rates
- `results/garcia/`: Garcia algorithm results for locally continuous kernels
- `results/cff/`: CFF method results for continuous-time systems

Example result format includes:
- Bound on expected coalescence time
- Empirical mean and standard error
- Confidence intervals
- Tightness of bounds

See [Perfect Simulations/src/simulators/README.md](Perfect%20Simulations/src/simulators/README.md) for detailed algorithm documentation.

## Usage Examples

See the individual simulation scripts in `Perfect Simulations/` for complete working examples:
- `simulate_gallo.py` - Complete Gallo implementation example
- `simulate_garcia.py` - Complete Garcia implementation example
- `simulate_cff.py` - Complete CFF implementation example

## Contributing

To add a new perfect simulation algorithm:

1. Extend the `PerfectSimulator` base class in `src/simulators/perfect_simulator.py`
2. Implement the `perfect_sample()` method
3. Add tests in the appropriate test directory
4. Document your algorithm in `src/simulators/README.md`

## References

- Gallo, A. (2009). Perfect Simulation of Markov chains
- Garcia, N. L. (2011). Perfect simulation for measure-valued processes
- Kendall, W. S., & Møller, J. (2000). Perfect simulation using dominating processes

## License

This project is available under the MIT License. See LICENSE file for details.

## Author

**Developed by:** AmTuTi1999

For questions or issues, please open an issue on GitHub.