# Garcia Perfect Simulation Implementation

## Overview

This implementation translates the Garcia (2011) perfect simulation algorithm from the mathematical description (LaTeX document) into Python code, following the structure of the existing Gallo implementation.

## Files

1. **`garcia_simulator.py`** - Main implementation of Garcia's algorithm
2. **`compare_gallo_garcia.py`** - Comparison between Gallo and Garcia approaches

## Key Components

### 1. GarciaContextTreeSimulator Class

The main class implementing Garcia's perfect simulation framework for locally continuous chains.

#### Core Methods

**Context Tree Operations:**
- `find_context()` - Find c_τ(past) = longest suffix in skeleton τ
- `context_length_at_time()` - Compute c_τ^n = sup_a |c_τ(Y_{-∞}^n(a))|

**Auxiliary Process:**
- `compute_Y_n()` - Generate Y_n from U_n via interval splitting (Eq 3.2)
- `generate_Y_sequence()` - Generate full Y sequence from U sequence

**Good Coalescence Times (Definition 3.3):**
- `is_good_coalescence_time()` - Check if time i satisfies: ∀j ∈ [m,n], Y_j ∈ A OR c_τ^j ≤ j - i
- `compute_bar_theta()` - Compute θ̄[m,n] = sup{i ≤ m : i is good}

**Block Construction (Section 3.4):**
- `construct_blocks()` - Build blocks {B_k} via θ^k recursion:
  - θ^{-1} := 1
  - θ^k = θ̄[θ^{k-1} - 1]
  - B_k = {θ^k, ..., θ^{k-1} - 1}

**Coalescence Time Computation:**
- `compute_zeta_i()` - Compute ζ_i = 𝟙{U_i ≥ α_{-1}} · Σ_k k · 𝟙{U_i ∈ [α_{k-1}, α_k)}
- `compute_L_k()` - Compute L_k = sup_{i ∈ B_k} ζ_i
- `compute_Theta()` - Compute Θ[k] = sup{n ≤ k : L_i ≤ i - n, ∀i = n,...,k}
- `compute_Lambda()` - Compute Λ[0] = -Σ_{i=0}^{-Θ[0]} |B_i|

**Update Function:**
- `interval_length()` - Compute |I_k(a|a_{-k}^{-1})|
- `update_function_F()` - Apply F(U_0, past) using interval splitting

**Main Algorithm:**
- `perfect_simulation()` - **Algorithm 1** from the document

## Algorithm Flow (Algorithm 1)

```
1. Generate U_i ~ Uniform[0,1)
2-6. Construct blocks:
   - Generate Y sequence
   - Compute context lengths c_τ^n
   - Build blocks via θ^k recursion
7-12. Compute L_k for each block
15-16. Compute Θ[n] and Λ[n]
19-25. Forward construction from Λ[n] to n
26. Return X_n
```

## Key Differences: Gallo vs Garcia

| Aspect | Gallo (2009) | Garcia (2011) |
|--------|-------------|---------------|
| **Coalescence** | Distance to reference string w | Good coalescence times θ̄[m] |
| **Context** | Via lag function l_w(k) | Via skeleton τ membership |
| **Structure** | No blocks | Blocks B_k via recursion |
| **Independence** | N/A | {L_k} are i.i.d. |
| **Truncation** | User limit S | No truncation |
| **Bias** | Biased if S < E[L_n] | Unbiased (exact) |
| **Complexity** | O(S) | O(Λ[0]) |

## Theoretical Advantages of Garcia

1. **True Perfect Simulation**: No truncation bias, samples exactly from stationary distribution μ

2. **Block Independence**: {L_k}_{k≥0} are i.i.d., enabling renewal-like analysis

3. **Feasibility Conditions** (Theorem 3.5):
   - (i) Σ_{k≥1} Π_{j=0}^{k-1} A_k = +∞ ⟹ Λ[0] a.s. finite
   - (ii) Σ_{k≥0} (1 - A_k) < +∞ ⟹ Λ[0] summable tail
   - (iii) Exponential decay ⟹ Λ[0] exponential tail

4. **Generality**: Works for arbitrary locally continuous kernels, not just specific models

## Usage Example

```python
from garcia_simulator import create_simple_garcia_simulator

# Create simulator
sim = create_simple_garcia_simulator()

# Generate perfect sample
X_n, info = sim.perfect_simulation(target_time=0)

print(f"Sample: {X_n}")
print(f"Coalescence time: {info['Lambda_n']}")
print(f"Number of blocks: {info['num_blocks']}")
```

## Mathematical Background

### Local Continuity Definition

A kernel P belongs to LC(τ) if for any v ∈ τ with |v| < +∞:

```
α_k^v := inf_{a_{-k}^{-1}} Σ_a inf_z P(a|v a_{-k}^{-1} z) → 1 as k → ∞
```

### Probabilistic Skeleton

The pair (τ, p) where:
- τ = context tree
- p(a|v) := inf_z P(a|vz) for v ∈ τ with |v| < +∞

### Good Coalescence Time

Time i is "good" for window [m,n] if:
```
∀j ∈ [m,n]: Y_j ∈ A  OR  c_τ^j ≤ j - i
```

This ensures that by time i, all future updates don't depend on the infinite past.

## Implementation Notes

1. **Simplified α sequences**: The implementation uses placeholder α sequences that should be replaced with actual kernel-specific values

2. **Block limits**: Set `max_blocks` and `max_block_size` to prevent infinite loops in practice

3. **Context detection**: Uses suffix matching to find c_τ(past) efficiently

4. **Forward construction**: Builds the sequence from Λ[0] to target time using the update function F

## References

- Garcia (2011): "Perfect simulation and finitary coding for locally continuous chains"
- Gallo (2009): "Perfect simulation for locally continuous chains with unbounded variable-length memory"
- The comparison script provides detailed algorithmic and theoretical comparisons

## Running the Code

```bash
# Run comparison
python compare_gallo_garcia.py

# Use in your code
from garcia_simulator import GarciaContextTreeSimulator
```

---

**Note**: This implementation follows the theoretical framework closely. For production use, the α sequences and transition probabilities should be computed from the actual kernel P being simulated.
