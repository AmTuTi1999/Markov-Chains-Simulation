# """
# Unified Experiment Runner: Gallo vs Garcia Perfect Simulation

# This script runs both Gallo and Garcia experiments and generates comparison plots.
# """

# import os
# import sys
# import numpy as np

# # Add src to path (adjust if needed)
# # sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# from run_garcia_experiments import (
#     run_garcia_simulation, 
#     create_simple_skeleton,
#     compare_garcia_vs_gallo
# )

# # Uncomment if you have the Gallo experiment runner
# # from run_gallo_experiments import run_gallo_simulation


# def run_unified_experiments(
#     run_gallo: bool = True,
#     run_garcia: bool = True,
#     compare_results: bool = True,
#     num_validation_samples: int = 1000,
# ):
#     """
#     Run both Gallo and Garcia experiments and compare results.
    
#     Args:
#         run_gallo: If True, run Gallo experiments
#         run_garcia: If True, run Garcia experiments
#         compare_results: If True, generate comparison plots
#         num_validation_samples: Number of samples for validation
#     """
    
#     results = {}
    
#     # ========================================================
#     # SETUP PARAMETERS
#     # ========================================================
    
#     # Common parameters
#     alphabet = [0, 1]
#     epsilon = 0.1
#     alpha_range = np.linspace(0.1, 0.5, 5)
    
#     print("=" * 80)
#     print("UNIFIED EXPERIMENT: GALLO vs GARCIA")
#     print("=" * 80)
#     print(f"Alphabet: {alphabet}")
#     print(f"Epsilon: {epsilon}")
#     print(f"Alpha range: {alpha_range}")
#     print(f"Validation samples: {num_validation_samples}")
#     print("=" * 80 + "\n")
    
#     # ========================================================
#     # RUN GALLO EXPERIMENTS
#     # ========================================================
    
#     if run_gallo:
#         print("\n" + "="*80)
#         print("RUNNING GALLO EXPERIMENTS")
#         print("="*80 + "\n")
        
#         # Gallo-specific parameters
#         reference_string = [1, 1]  # w = [1, 1]
#         beta = 0.7  # AR decay exponent
        
#         gallo_args = {
#             'alphas': list(alpha_range),
#             'alphabet': alphabet,
#             'reference_string': reference_string,
#             'epsilon': epsilon,
#             'beta': beta,
#         }
        
#         try:
#             # Uncomment if you have the Gallo experiment runner
#             # results['gallo'] = run_gallo_simulation(
#             #     window=(0, 0),
#             #     args=gallo_args,
#             #     max_trie_depth=8,
#             #     max_depth=50,
#             #     num_validation_samples=num_validation_samples,
#             #     validate_with_simulation=True,
#             #     run_truncated=True,
#             #     run_non_truncated=False,
#             # )
            
#             print("NOTE: Gallo experiment runner not imported.")
#             print("      To enable, uncomment the import and function call above.")
#             results['gallo'] = None
            
#         except Exception as e:
#             print(f"Error running Gallo experiments: {e}")
#             results['gallo'] = None
    
#     # ========================================================
#     # RUN GARCIA EXPERIMENTS
#     # ========================================================
    
#     if run_garcia:
#         print("\n" + "="*80)
#         print("RUNNING GARCIA EXPERIMENTS")
#         print("="*80 + "\n")
        
#         # Create skeleton for Garcia
#         skeleton, _ = create_simple_skeleton()
        
#         garcia_args = {
#             'skeleton': skeleton,
#             'alphabet': alphabet,
#             'alpha_sequence': {},  # Generated in function
#             'epsilon': epsilon,
#             'alpha_params': list(alpha_range),
#         }
        
#         try:
#             results['garcia'] = run_garcia_simulation(
#                 window=(0, 0),
#                 args=garcia_args,
#                 max_blocks=100,
#                 max_block_size=50,
#                 num_validation_samples=num_validation_samples,
#                 validate_with_simulation=True,
#                 run_coalescence_analysis=True,
#             )
#         except Exception as e:
#             print(f"Error running Garcia experiments: {e}")
#             results['garcia'] = None
    
#     # ========================================================
#     # GENERATE COMPARISON
#     # ========================================================
    
#     if compare_results and results.get('garcia') and results.get('gallo'):
#         print("\n" + "="*80)
#         print("GENERATING COMPARISON PLOTS")
#         print("="*80 + "\n")
        
#         results_dir = os.path.join("results", "comparison")
#         os.makedirs(results_dir, exist_ok=True)
        
#         filename_suffix = f"unified_alpha_{alpha_range[0]:.2f}_to_{alpha_range[-1]:.2f}"
        
#         try:
#             compare_garcia_vs_gallo(
#                 garcia_results=results['garcia'],
#                 gallo_results=results['gallo'],
#                 results_dir=results_dir,
#                 filename_suffix=filename_suffix
#             )
#         except Exception as e:
#             print(f"Error generating comparison: {e}")
    
#     # ========================================================
#     # SUMMARY
#     # ========================================================
    
#     print("\n" + "="*80)
#     print("EXPERIMENT SUMMARY")
#     print("="*80)
    
#     if results.get('gallo'):
#         print("\n✓ Gallo experiments completed")
#         print(f"  Results saved in: results/gallo/")
#     elif run_gallo:
#         print("\n✗ Gallo experiments failed or not available")
    
#     if results.get('garcia'):
#         print("\n✓ Garcia experiments completed")
#         print(f"  Results saved in: results/garcia/")
#     elif run_garcia:
#         print("\n✗ Garcia experiments failed")
    
#     if compare_results and results.get('garcia') and results.get('gallo'):
#         print("\n✓ Comparison plots generated")
#         print(f"  Results saved in: results/comparison/")
    
#     print("\n" + "="*80)
#     print("THEORETICAL COMPARISON SUMMARY")
#     print("="*80)
    
#     print("""
#     GALLO (2009):
#     - Uses lookback depth L_n based on distance to reference string w
#     - Truncation at user limit S introduces bias
#     - Simple conceptually, but E[L̃_n] ≤ E[L_n]
#     - No block independence structure
    
#     GARCIA (2011):
#     - Uses blocked rescaled coalescence time Λ[0]
#     - True perfect simulation (no truncation bias)
#     - Block structure: {L_k}_{k≥0} are i.i.d. (KEY PROPERTY!)
#     - Renewal-like analysis enables tight tail bounds
#     - Feasibility conditions (Theorem 3.5):
#         (i)   Σ Π A_k = +∞  →  Λ[0] is a.s. finite
#         (ii)  Σ(1-A_k) < +∞  →  Λ[0] has summable tail
#         (iii) Exponential decay  →  Λ[0] has exponential tail
    
#     TRADE-OFF:
#     - Gallo: Simpler algorithm, faster, but biased if S < E[L_n]
#     - Garcia: More complex, but exact sampling from stationary distribution
#     """)
    
#     print("="*80 + "\n")
    
#     return results


# def generate_theoretical_comparison_document(output_dir: str = "results"):
#     """
#     Generate a comprehensive theoretical comparison document.
    
#     Args:
#         output_dir: Directory to save the document
#     """
    
#     doc_path = os.path.join(output_dir, "theoretical_comparison.txt")
    
#     with open(doc_path, 'w') as f:
#         f.write("=" * 80 + "\n")
#         f.write("THEORETICAL COMPARISON: GALLO vs GARCIA PERFECT SIMULATION\n")
#         f.write("=" * 80 + "\n\n")
        
#         f.write("1. COALESCENCE MECHANISMS\n")
#         f.write("-" * 80 + "\n")
#         f.write("GALLO:\n")
#         f.write("  L_n = m_n + |w| + l_w(m_n)\n")
#         f.write("  where m_n = distance to last occurrence of reference string w\n")
#         f.write("  and l_w(k) = exp(α·k) is the lag function\n\n")
        
#         f.write("GARCIA:\n")
#         f.write("  Λ[0] = -Σ_{i=0}^{-Θ[0]} |B_i|\n")
#         f.write("  where Θ[0] is the blocked rescaled coalescence time\n")
#         f.write("  and B_i are blocks defined via good coalescence times\n\n")
        
#         f.write("2. KEY STRUCTURAL DIFFERENCES\n")
#         f.write("-" * 80 + "\n")
#         f.write(f"{'Property':<30} {'Gallo':<25} {'Garcia':<25}\n")
#         f.write("-" * 80 + "\n")
        
#         properties = [
#             ("Coalescence Detection", "Via reference string", "Via good coal. times"),
#             ("Context Tree", "Generated via l_w(k)", "Pre-specified skeleton τ"),
#             ("Block Structure", "None", "Recursive blocks B_k"),
#             ("Independence", "No", "Yes: {L_k} are i.i.d."),
#             ("Truncation", "Required (user limit S)", "Not needed"),
#             ("Bias", "Yes (if S < E[L_n])", "No (exact sampling)"),
#             ("Analysis Tool", "Direct PMF", "Renewal theory"),
#             ("Tail Bounds", "Geometric mixing", "Block independence"),
#         ]
        
#         for prop, gallo, garcia in properties:
#             f.write(f"{prop:<30} {gallo:<25} {garcia:<25}\n")
        
#         f.write("\n3. MATHEMATICAL FOUNDATIONS\n")
#         f.write("-" * 80 + "\n")
        
#         f.write("\nGALLO - Local Continuity via Reference String:\n")
#         f.write("  Definition: For each v, the sequence α_k^v converges to 1\n")
#         f.write("  Construction: τ = ⋃_{i≥0} ⋃_{c∈A^{l_w(i)}} c·w·A^i\n")
#         f.write("  Lookback: L_n depends on distance to last w occurrence\n")
#         f.write("  Truncation: L̃_n = min(L_n, S) introduces bias\n\n")
        
#         f.write("GARCIA - Good Coalescence Times:\n")
#         f.write("  Definition: i is 'good' for [m,n] if ∀j ∈ [m,n]:\n")
#         f.write("             Y_j ∈ A OR c_τ^j ≤ j - i\n")
#         f.write("  Block Construction:\n")
#         f.write("    θ^{-1} := 1\n")
#         f.write("    θ^k = θ̄[θ^{k-1} - 1]\n")
#         f.write("    B_k = {θ^k, ..., θ^{k-1} - 1}\n")
#         f.write("  Key Property: {L_k := sup_{i∈B_k} ζ_i}_{k≥0} are i.i.d.\n\n")
        
#         f.write("4. FEASIBILITY CONDITIONS\n")
#         f.write("-" * 80 + "\n")
        
#         f.write("\nGALLO:\n")
#         f.write("  E[L_n] < ∞ (sufficient for almost sure finiteness)\n")
#         f.write("  Typically requires: α < log(1/p_w) where p_w = ε^|w|\n\n")
        
#         f.write("GARCIA (Theorem 3.5):\n")
#         f.write("  Let A_k = {1 - (E|θ̄[0]| + 1)P(U_0 > α_k^{c_τ^{-1}})} ∨ α_{-1}\n")
#         f.write("  \n")
#         f.write("  (i)   Σ_{k≥1} Π_{j=0}^{k-1} A_k = +∞\n")
#         f.write("        ⟹ Λ[0] is a.s. finite\n")
#         f.write("  \n")
#         f.write("  (ii)  Σ_{k≥0} (1 - A_k) < +∞\n")
#         f.write("        ⟹ Λ[0] has summable tail\n")
#         f.write("  \n")
#         f.write("  (iii) θ̄[0] has exponential tail AND {1-A_k} decays exponentially\n")
#         f.write("        ⟹ Λ[0] has exponential tail\n\n")
        
#         f.write("5. ADVANTAGES AND DISADVANTAGES\n")
#         f.write("-" * 80 + "\n")
        
#         f.write("\nGALLO Advantages:\n")
#         f.write("  + Simple algorithm (just lookback)\n")
#         f.write("  + Easy to implement\n")
#         f.write("  + Fast (O(S) per sample)\n")
#         f.write("  + Direct analytical formulas for E[L_n]\n")
#         f.write("  + Works well when S ≥ E[L_n]\n\n")
        
#         f.write("GALLO Disadvantages:\n")
#         f.write("  - User must choose truncation limit S\n")
#         f.write("  - Biased if S < E[L_n]\n")
#         f.write("  - User impatience: P(L_n > S) / P(L_n ≤ S)\n")
#         f.write("  - No block independence for analysis\n")
#         f.write("  - Specific to models with reference strings\n\n")
        
#         f.write("GARCIA Advantages:\n")
#         f.write("  + True perfect simulation (exact, unbiased)\n")
#         f.write("  + Block independence: {L_k} are i.i.d.\n")
#         f.write("  + Renewal-like structure\n")
#         f.write("  + Rigorous tail bounds\n")
#         f.write("  + General framework (any locally continuous kernel)\n")
#         f.write("  + No user-specified parameters (S, w, etc.)\n\n")
        
#         f.write("GARCIA Disadvantages:\n")
#         f.write("  - More complex algorithm\n")
#         f.write("  - Harder to implement correctly\n")
#         f.write("  - Computational cost O(Λ[0])\n")
#         f.write("  - Block construction overhead\n")
#         f.write("  - Less intuitive than lookback\n\n")
        
#         f.write("6. WHEN TO USE WHICH?\n")
#         f.write("-" * 80 + "\n")
        
#         f.write("\nUse GALLO when:\n")
#         f.write("  • You have a natural reference string w\n")
#         f.write("  • You can afford truncation bias\n")
#         f.write("  • Speed is critical\n")
#         f.write("  • E[L_n] is well-understood and small\n")
#         f.write("  • Approximate sampling is acceptable\n\n")
        
#         f.write("Use GARCIA when:\n")
#         f.write("  • Exact sampling is required\n")
#         f.write("  • You need provable unbiasedness\n")
#         f.write("  • Working with general locally continuous kernels\n")
#         f.write("  • Block independence is useful for analysis\n")
#         f.write("  • Rigorous tail bounds are needed\n\n")
        
#         f.write("7. RELATIONSHIP BETWEEN THE METHODS\n")
#         f.write("-" * 80 + "\n")
#         f.write("\nGarcia generalizes Gallo:\n")
#         f.write("  • Gallo's construction can be viewed as a special case\n")
#         f.write("  • Garcia removes the truncation requirement\n")
#         f.write("  • Garcia provides stronger theoretical guarantees\n")
#         f.write("  • Both achieve perfect simulation, but Garcia is exact\n\n")
        
#         f.write("Expected Relationship:\n")
#         f.write("  E[|Λ[0]|] ≥ E[L̃_n] (Garcia ≥ truncated Gallo)\n")
#         f.write("  but E[|Λ[0]|] should be close to E[L_n] (true Gallo expectation)\n\n")
        
#         f.write("=" * 80 + "\n")
#         f.write("CONCLUSION\n")
#         f.write("=" * 80 + "\n")
#         f.write("\nBoth methods achieve perfect simulation, but with different trade-offs:\n\n")
#         f.write("GALLO: Fast, simple, intuitive, but potentially biased\n")
#         f.write("GARCIA: Complex, rigorous, exact, but computationally heavier\n\n")
#         f.write("The choice depends on the application requirements and whether\n")
#         f.write("exact unbiased sampling justifies the additional complexity.\n")
#         f.write("=" * 80 + "\n")
    
#     print(f"Saved theoretical comparison to: {doc_path}")


# if __name__ == "__main__":
#     # Run unified experiments
#     results = run_unified_experiments(
#         run_gallo=True,  # Set to True if you have Gallo experiment runner
#         run_garcia=True,
#         compare_results=True,
#         num_validation_samples=1000,
#     )
    
#     # Generate theoretical comparison document
#     generate_theoretical_comparison_document()
    
#     print("\n" + "="*80)
#     print("ALL EXPERIMENTS COMPLETED")
#     print("="*80)
