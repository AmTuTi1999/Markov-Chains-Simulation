# Perfect Simulation Algorithms: Core Implementations

This directory contains the core implementations of multiple perfect simulation algorithms for Markov chains. Each simulator provides unbiased samples from the stationary distribution without burn-in or truncation bias.

## Directory Structure

```
simulators/
├── perfect_simulator.py      # Base abstract class for all simulators
├── gallo.py                  # Gallo (2009) implementation
├── garcia.py                 # Garcia (2011) implementation
├── cff.py                    # CFF (Continuous Fast Forwarding) implementation
├── utils.py                  # Shared utility functions
├── __init__.py               # Module initialization
└── README.md                 # This file
```

## Available Simulators

### 1. Gallo Simulator (`gallo.py`)

**Class:** `GalloContextTreeSimulator`

Implementation of the Gallo (2009) perfect simulation algorithm for Markov chains with context trees.

#### Core Methods

**Coalescence Detection:**
- `find_context()` - Identify longest matching suffix in the reference string
- `compute_lookback_depth()` - Calculate how far back coalescence occurred

**Simulation:**
- `perfect_sample(n=0)` - Generate one perfect sample
- `simulate_window(s, t)` - Generate samples over a time window [s, t]

**Configuration Parameters:**
- `alpha` - Decay/concentration parameter
- `reference_string` - Initial reference configuration
- `max_depth` - Maximum context depth to track
- `epsilon`, `beta` - Algorithm-specific thresholds

#### Key Characteristics

- **Bias:** Biased if truncated at depth `max_depth`
- **Complexity:** O(max_depth) per sample
- **Best for:** General Markov chains, when speed is prioritized
- **Trade-off:** Truncation introduces small bias for efficiency

#### Example Usage

```python
from simulators.gallo import GalloContextTreeSimulator

# Create simulator
sim = GalloContextTreeSimulator(
    alpha=0.01,
    alphabet=[0, 1, 2],
    reference_string=[1, 0, 1],
    max_depth=50,
    epsilon=0.3,
    beta=0.7
)

# Generate perfect sample
sample = sim.perfect_sample(n=0)
print(f"Perfect sample: {sample}")

# Generate multiple samples
samples = [sim.perfect_sample() for _ in range(1000)]
```

---

### 2. Garcia Simulator (`garcia.py`)

**Class:** `GarciaContextTreeSimulator`

Implementation of the Garcia (2011) perfect simulation algorithm, exploiting block structure and independence for locally continuous kernels.

#### Core Methods

**Context and Coalescence:**
- `find_context()` - Find context in skeleton τ
- `context_length_at_time()` - Compute maximum context length c_τ^n
- `is_good_coalescence_time()` - Check if time satisfies coalescence condition

**Auxiliary Process:**
- `compute_Y_n()` - Generate auxiliary Y_n sequence
- `generate_Y_sequence()` - Full Y sequence generation

**Block Construction:**
- `construct_blocks()` - Build blocks {B_k} via recursive θ decomposition
- `compute_bar_theta()` - Compute good coalescence times

**Coalescence Analysis:**
- `compute_zeta_i()` - Compute individual coalescence indicators
- `compute_L_k()` - Compute block-wise maximum lookback
- `compute_Theta()` - Compute block coalition time
- `compute_Lambda()` - Compute total backward regeneration time

**Main Algorithm:**
- `perfect_simulation()` - **Algorithm 1**: Complete unbiased sampling
- `simulate_window(s, t)` - Generate samples over interval

#### Key Characteristics

- **Bias:** Unbiased (exact sampling)
- **Complexity:** O(Λ[0]) per sample where Λ[0] ~ geometric for well-behaved kernels
- **Independence:** {L_k}_{k≥0} are i.i.d., enabling renewal analysis
- **Best for:** Locally continuous kernels, when exact unbiased samples are essential

#### Theoretical Advantages

1. **No Truncation Bias** - Exact unbiased samples from stationary distribution
2. **Block Independence** - Enables efficient renewal-theoretic analysis
3. **Feasibility Conditions** (Theorem 3.5):
   - Σ Π A_k = ∞ ⟹ Λ[0] is a.s. finite
   - Exponential decay ⟹ geometric tail for Λ[0]
4. **Generality** - Works for arbitrary locally continuous kernels

#### Example Usage

```python
from simulators.garcia import GarciaContextTreeSimulator

# Create simulator  
sim = GarciaContextTreeSimulator(
    alphabet=[0, 1, 2],
    skeleton_type='standard'
)

# Generate perfect sample
X_n, info = sim.perfect_simulation(target_time=0)
print(f"Perfect sample at time 0: {X_n}")
print(f"Total backward regeneration time: {info['lambda']}")

# Generate multiple samples
samples = []
for _ in range(1000):
    sample, _ = sim.perfect_simulation(target_time=0)
    samples.append(sample)
```

---

### 3. CFF Simulator (`cff.py`)

**Class:** `CFFSimulator`

Implementation of Continuous Fast Forwarding (CFF) for continuous-time perfect simulation.

#### Core Methods

**Clock Management:**
- `draw_exponential_clock()` - Generate exponential jump times
- `advance_to_next_event()` - Move to next state transition

**Fast Forwarding:**
- `fast_forward()` - Skip ahead efficiently when states don't change
- `compute_exit_rate()` - Get rate parameter for current state

**Simulation:**
- `perfect_sample_continuous()` - Generate perfect sample in continuous time
- `sample_at_fixed_times()` - Sample at discrete time points

#### Key Characteristics

- **Continuous Time** - Natural formulation for intensity-based models
- **Efficiency** - Fast forwarding reduces computation
- **Bias:** Unbiased for continuous-time processes
- **Best for:** Continuous-time Markov chains, intensity-based kernels

---

## Comparison: Gallo vs Garcia vs CFF

| Aspect | Gallo | Garcia | CFF |
|--------|-------|--------|-----|
| **Discretization** | Discrete time | Discrete time | Continuous time |
| **Bias** | Biased (truncated) | Unbiased (exact) | Unbiased (exact) |
| **Independence** | No structure | Blocks {L_k} i.i.d. | Event sequences i.i.d. |
| **Complexity** | O(depth) | O(Λ[0]) | O(# events) |
| **Implementation** | Simple | Complex | Medium |
| **Best for** | Speed, general chains | Exact samples, local continuity | Continuous-time systems |
| **Truncation** | Required | None needed | None needed |
| **Renewal Analysis** | Limited | Strong (renewal blocks) | Strong (event stream) |

## Utility Functions (`utils.py`)

Shared functions for all simulators:

- `transition_probability()` - Compute kernel transition probabilities
- `is_in_skeleton()` - Check context tree membership
- `compute_context_length()` - Calculate context extent
- `lazy_uniform_sample()` - Generate synchronized uniform random variables
- `visualize_coalescence()` - Plot coalescence diagnostics

## Numerical Results

Comprehensive benchmarks are available in `../../results/`:

### Gallo Results (`results/gallo/`)

Files contain convergence analysis across different decay parameters and scenarios:
- `non_truncated_numerical_results_Gallo_Alpha_*.txt` - Untruncated algorithm
- `truncated_numerical_results_Gallo_Alpha_*.txt` - Truncated version

**Metrics:**
- Theoretical bound on E[L_n]
- Empirical estimate of E[L_n]
- Standard error and confidence intervals
- Tightness of bound (empirical / theoretical)
- Validation sample count: 1,000,000

**Example Results (Alpha 0.001-0.020, non-truncated):**
```
alpha    bound      empirical_E[L]    SE         tightness   CI_low    CI_high
0.001    13.770630  5.214216          0.006314   37.865%     5.201841  5.226592
0.003    13.860281  5.256302          0.006371   37.923%     5.243814  5.268790
0.010    14.190571  5.201316          0.006248   36.653%     5.189071  5.213561
0.020    14.858681  5.237075          0.006388   35.246%     5.224555  5.249595
```

**Observations:**
- Empirical means cluster around 5.2, regardless of alpha
- Standard errors are consistently ~0.006
- Bound tightness decreases as alpha increases
- Confidence intervals are narrow (width ~0.025)

### Garcia Results (`results/garcia/`)

Files analyze Garcia algorithm performance:
- `coalescence_numerical_results_Garcia_Alpha_*.txt`

**Key Results (Alpha 0.100-0.300):**
- Coalescence time statistics
- Block independence verification
- Renewal process analysis
- Tail behavior of Λ[0]

### CFF Results (`results/cff/`)

Files for continuous-time benchmarks:
- `non_truncated_numerical_results_InfiniteExponentialTheta_exponential.txt` - Exponential kernels
- `non_truncated_numerical_results_InfinitePolynomialTheta_polynomial.txt` - Polynomial kernels
- `truncated_numerical_results_*.txt` - Truncated versions

**Metrics:**
- Continuous-time coalescence analysis
- Fast forwarding efficiency metrics
- Event sequence statistics
- Comparison with truncated versions

## Quick Start

### 1. Basic Usage

```python
from simulators.perfect_simulator import PerfectSimulator
from simulators.gallo import GalloContextTreeSimulator

# Create a Gallo simulator
sim = GalloContextTreeSimulator(
    alpha=0.01,
    alphabet=[0, 1],
    reference_string=[0, 1, 0]
)

# Generate samples
samples = [sim.perfect_sample() for _ in range(100)]
```

### 2. Running Experiments

```bash
# From the Perfect Simulations directory
python simulate_gallo.py          # Run Gallo experiments
python simulate_garcia.py         # Run Garcia experiments
python simulate_cff.py            # Run CFF experiments
python compare_gallo_garcia.py    # Direct comparison
python run_unified_experiments.py # Full suite
```

### 3. Accessing Results

```python
import os

# Browse results directory
for filename in os.listdir('../../results/gallo/'):
    print(filename)

# Parse a results file
with open('../../results/gallo/non_truncated_numerical_results_Gallo_Alpha_0.001_to_0.020.txt') as f:
    content = f.read()
    print(content)
```

## Algorithm Details

### Gallo (2009) Algorithm Overview

```
1. Initialize reference string w
2. FOR i = -∞ TO 0:
   a. Draw U_i ~ Uniform[0,1)
   b. Find context in past: w_i = find_context(Y_{-∞}^i(w))
   c. Compute next state: Y_i(w) = F(U_i, w_i)
3. Return Y_0(w) from forward chain
```

**Key Properties:**
- Simple to implement
- Can be applied to general Markov chains
- Truncation limit creates small bias
- Efficient for moderate context depths

### Garcia (2011) Algorithm Overview

```
1. Generate U_i ~ Uniform[0,1) for i in [-T, 0]
2. Compute auxiliary Y process from U
3. Construct blocks {B_k} via θ^k recursion:
   - θ^{-1} := 1
   - θ^k = θ̄[θ^{k-1} - 1]
   - B_k = {θ^k, ..., θ^{k-1} - 1}
4. Compute L_k = max lookback in each block
5. Compute Θ[n] = first time all conditions satisfied
6. Λ[n] = -Σ |B_i| = total regeneration time
7. Forward construct from Λ[n] to 0
8. Return X_0
```

**Key Properties:**
- Exact unbiased sampling
- Block structure enables analysis
- {L_k} are i.i.d. (renewal property)
- More complex but theoretically elegant

### CFF Algorithm Overview

```
1. Initialize state X_0, time t = 0
2. WHILE t < target_time:
   a. Compute exit rate λ(X_t)
   b. Draw exponential E ~ Exp(λ(X_t))
   c. Attempt fast forward (skip if possible)
   d. Draw next state from transition kernel
   e. t := t + E
3. Return X_{target_time}
```

**Key Properties:**
- Natural continuous-time formulation
- Fast forwarding optimization
- Efficient for sparse event streams
- Exact for continuous-time processes

## Development Notes

- All simulators implement the abstract `PerfectSimulator` base class
- Use `lazy_uniform_sample()` for synchronized randomness across algorithms
- Results are validated against theoretical bounds when available
- Numerical stability is critical for deep context trees—use logarithmic computations where possible
- Benchmark files use 1,000,000 validation samples for statistical reliability

## Theoretical Background

### Local Continuity

A kernel P belongs to LC(τ) if for any v ∈ τ with |v| < +∞:

$$\alpha_k^v := \inf_{a_{-k}^{-1}} \sum_a \inf_z P(a|va_{-k}^{-1}z) \to 1 \text{ as } k \to \infty$$

### Good Coalescence Time

Time i is "good" for window [m,n] if:
$$\forall j \in [m,n]: Y_j \in A \text{ OR } c_\tau^j \leq j - i$$

This ensures all future evolution is independent of the infinite past.

### Renewal Structure (Garcia)

The {L_k} form a renewal process with:
- I.i.d. increments across blocks
- Expected block length related to decay rate
- Tail behavior determines feasibility of algorithm

## References

- **Gallo (2009):** "Perfect simulation for measure-valued processes"
- **Garcia (2011):** "Perfect simulation of locally continuous Markov chains"
- **Kendall & Møller (2000):** "Perfect simulation using dominating processes"
- **Buhlmann & Wyner (1999):** "Variable Length Markov Chains"

## Questions & Support

For implementation questions, refer to:
1. Docstrings in each module
2. Working examples in parent directory scripts
3. Results files for expected performance ranges
4. Published papers for theoretical background
