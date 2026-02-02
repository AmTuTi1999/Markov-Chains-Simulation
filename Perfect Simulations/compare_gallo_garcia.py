"""
Comparison between Gallo and Garcia Perfect Simulation Approaches

Key Differences:
================

GALLO (2009):
- Direct lookback depth L_n
- Based on distance to last occurrence of reference string w
- Truncation at user-imposed limit S
- Lookback: L_n = m_n + |w| + l_w(m_n)

GARCIA (2011):
- Blocked rescaled coalescence time Λ[0]
- Based on good coalescence times θ̄[m]
- Block construction via θ^k recursion
- Update function with interval splitting

Both achieve perfect simulation (samples from stationary distribution μ).
"""

from src.simulators.garcia import create_simple_garcia_simulator

def compare_approaches():
    """
    Compare Gallo vs Garcia on a simple example.
    """
    print("=" * 80)
    print("COMPARISON: GALLO vs GARCIA PERFECT SIMULATION")
    print("=" * 80)
    
    print("\n" + "─" * 80)
    print("GALLO APPROACH (2009)")
    print("─" * 80)
    print("""
    1. Define reference string w (e.g., w = [-1, +1])
    2. Generate context tree τ via lag function l_w(k) = exp(α·k)
    3. For each time n, compute lookback depth:
       
       L_n = {  0                        if U_n < #A·ε  (spontaneous)
             {  m_n + |w| + l_w(m_n)      otherwise
       
       where m_n = distance to last occurrence of w
    
    4. Truncate at user limit S: L̃_n = min(L_n, S)
    5. Sample X_n using past X_{n-L̃_n}^{n-1}
    
    Advantages:
    - Simple conceptually (just lookback)
    - Direct truncation strategy
    - Easy to compute E[L_n] analytically
    
    Disadvantages:
    - User impatience bias when S < ∞
    - No independence between blocks
    - Truncation affects distribution
    """)
    
    print("\n" + "─" * 80)
    print("GARCIA APPROACH (2011)")
    print("─" * 80)
    print("""
    1. Generate auxiliary process Y_n from U_n
    2. Compute context lengths c_τ^n at each time
    3. Find good coalescence times:
       
       i is "good" for [m,n] if ∀j ∈ [m,n]:
           Y_j ∈ A  OR  c_τ^j ≤ j - i
    
    4. Construct blocks via recursion:
       θ^{-1} := 1
       θ^k = θ̄[θ^{k-1} - 1]  (max good coalescence time)
       B_k = {θ^k, ..., θ^{k-1} - 1}
    
    5. Compute L_k = sup_{i ∈ B_k} ζ_i (max context depth in block)
    6. Find Θ[0] = sup{n ≤ 0 : L_i ≤ i - n, ∀i = n,...,0}
    7. Coalescence time: Λ[0] = -Σ_{i=0}^{-Θ[0]} |B_i|
    8. Forward construction from Λ[0] to 0
    
    Advantages:
    - True perfect simulation (no truncation bias)
    - Blocks {L_k} are i.i.d. (key property!)
    - Renewal-like structure for analysis
    - Feasibility conditions (Theorem 3.5)
    
    Disadvantages:
    - More complex algorithm
    - Requires careful block construction
    - Harder to compute E[Λ[0]] analytically
    """)
    
    print("\n" + "─" * 80)
    print("KEY THEORETICAL DIFFERENCES")
    print("─" * 80)
    
    comparison = {
        'Property': [
            'Coalescence mechanism',
            'Context detection',
            'Block structure',
            'Independence',
            'Truncation',
            'Bias',
            'Convergence',
            'Complexity'
        ],
        'Gallo (2009)': [
            'Distance to reference string w',
            'Via lag function l_w(k)',
            'No block structure',
            'Not applicable',
            'User-imposed limit S',
            'Bias when S < E[L_n]',
            'E[L̃_n] ≤ E[L_n]',
            'O(S) per sample'
        ],
        'Garcia (2011)': [
            'Good coalescence times θ̄[m]',
            'Via skeleton τ membership',
            'Blocks B_k via θ^k recursion',
            '{L_k} are i.i.d.',
            'No truncation needed',
            'Unbiased (exact stationary)',
            'E[Λ[0]] finite under conditions',
            'O(Λ[0]) per sample'
        ]
    }
    
    for i, prop in enumerate(comparison['Property']):
        print(f"\n{prop}:")
        print(f"  Gallo:  {comparison['Gallo (2009)'][i]}")
        print(f"  Garcia: {comparison['Garcia (2011)'][i]}")
    
    print("\n" + "─" * 80)
    print("FEASIBILITY CONDITIONS")
    print("─" * 80)
    
    print("""
    GARCIA THEOREM 3.5: CFTP is feasible if one of:
    
    (i)   Σ_{k≥1} Π_{j=0}^{k-1} A_k = +∞
          ⟹ Λ[0] is a.s. finite
    
    (ii)  Σ_{k≥0} (1 - A_k) < +∞
          ⟹ Λ[0] has summable tail
    
    (iii) θ̄[0] has exponential tail AND {1 - A_k} decays exponentially
          ⟹ Λ[0] has exponential tail
    
    where A_k = {1 - (E|θ̄[0]| + 1)P(U_0 > α_k^{c_τ^{-1}})} ∨ α_{-1}
    
    This generalizes Gallo's condition E[L_n] < ∞.
    """)
    
    print("\n" + "─" * 80)
    print("DEMONSTRATION: GARCIA SIMULATOR")
    print("─" * 80)
    
    # Create Garcia simulator
    sim = create_simple_garcia_simulator()
    
    print("\nGenerating sample from stationary distribution...")
    X_n, info = sim.perfect_simulation()
    
    print("\nResults:")
    print(f"  Sample X_0 = {X_n}")
    print(f"  Coalescence time Λ[0] = {info['Lambda_n']}")
    print(f"  Number of blocks = {info['num_blocks']}")
    print(f"  Block rescaled coalescence Θ[0] = {info['Theta_n']}")
    print(f"  L_k sequence = {info['L_sequence'][:5]}..." if len(info['L_sequence']) > 5 else f"  L_k sequence = {info['L_sequence']}")
    
    print("\n" + "─" * 80)
    print("CONCLUSION")
    print("─" * 80)
    print("""
    Garcia's approach extends Gallo by:
    1. Removing truncation bias (true perfect simulation)
    2. Establishing i.i.d. block structure {L_k}
    3. Providing general feasibility conditions
    4. Working with arbitrary locally continuous kernels
    
    Trade-off: More complex algorithm vs. exact sampling
    """)
    
    print("=" * 80)

if __name__ == "__main__":
    compare_approaches()
